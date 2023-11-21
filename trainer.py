import copy
import logging
import math
import os
import random
from typing import Dict, List, Tuple, Optional, Any, Union, Mapping

import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from transformers import get_linear_schedule_with_warmup, PreTrainedModel

from utils import AverageMeter, ProgressMeter

logger = logging.getLogger(__name__)


class CropSentenceTrainer(object):
    def __init__(self, model, train_dataset, data_collator, training_args, data_args, tokenizer,
                 use_small_batch_size=None):
        super(CropSentenceTrainer, self).__init__()
        self.training_args = training_args
        self.data_args = data_args
        self.args = training_args
        self.epoch = 0

        if dist.is_initialized() and dist.get_world_size() > 1:
            assert self.training_args.negatives_x_device, self.training_args.negatives_x_device
        self._dist_loss_scale_factor = dist.get_world_size() if self.training_args.negatives_x_device else 1

        if self.training_args.negatives_x_device:
            assert dist.is_initialized() and dist.get_world_size() > 1, \
                ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.use_small_batch_size = use_small_batch_size

        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.data_collator = data_collator

        if isinstance(self.train_dataset, list):
            self.train_dataloader = []
            for idx, dataset in enumerate(train_dataset):
                use_small_batch_size = self.use_small_batch_size[idx] if self.use_small_batch_size is not None \
                    else False
                self.train_dataloader.append(self.get_train_dataloader(dataset, use_small_batch_size))
        else:
            self.train_dataloader = self.get_train_dataloader(self.train_dataset)

        if isinstance(self.train_dataloader, list):
            assert training_args.multi_task, \
                ValueError('can only have multiple datasets when using multi-task learning')
            self.num_training_steps = 0
            for dataloader in self.train_dataloader:
                self.num_training_steps += len(dataloader) // self.training_args.gradient_accumulation_steps
        else:
            self.num_training_steps = len(self.train_dataloader) // self.training_args.gradient_accumulation_steps
        if self.training_args.max_steps > 0:
            if isinstance(self.train_dataloader, list):
                self.max_step = self.training_args.max_steps * len(self.train_dataloader)
            else:
                self.max_step = self.training_args.max_steps
            self.num_train_epochs = 1
        else:
            self.max_step = self.training_args.num_train_epochs * self.num_training_steps
            self.num_train_epochs = math.ceil(self.training_args.num_train_epochs)
        self.model = self.setup_model(model)
        self.optimizer = self.get_optimizer(self.model, self.training_args.weight_decay,
                                            self.training_args.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.training_args.warmup_ratio * self.max_step,
            num_training_steps=self.max_step
        )
        os.makedirs(self.training_args.output_dir, exist_ok=True)

        self.use_amp = False
        self.amp_dtype = False
        self.scaler = None
        if self.training_args.fp16 or self.training_args.bf16:
            self.use_amp = True
            self.amp_dtype = torch.float16 if self.training_args.fp16 else torch.bfloat16
            self.scaler = torch.cuda.amp.GradScaler()

    def setup_model(self, model):
        model = model.to(self.training_args.device)
        if self.training_args.n_gpu > 1:
            model = nn.DataParallel(model)
        elif self.training_args.local_rank != -1:
            kwargs = {}
            if self.training_args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.training_args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.training_args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.training_args.ddp_bucket_cap_mb
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.training_args.device] if self.training_args.n_gpu != 0 else None,
                output_device=self.training_args.device if self.training_args.n_gpu != 0 else None,
                broadcast_buffers=False,
                **kwargs,
            )
        return model

    def get_optimizer(self, model, weight_decay, lr):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        p.requires_grad and not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        adam_kwargs = {
            "betas": (self.training_args.adam_beta1, self.training_args.adam_beta2),
            "eps": self.training_args.adam_epsilon,
        }
        return AdamW(optimizer_grouped_parameters, lr=lr, **adam_kwargs)

    def _save(self, model_to_save, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save
        if isinstance(model_to_save, PreTrainedModel):
            model_to_save.save_pretrained(output_dir)
        else:
            model_to_save.save(output_dir)

    def save_model(self):
        if self.is_world_process_zero():
            self._save(self.model)

    def is_world_process_zero(self) -> bool:
        return self.training_args.process_index == 0

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(self.training_args.device)
        return data

    def _prepare_inputs(
            self,
            inputs: Union[Tuple[Dict[str, Union[torch.Tensor, Any]], ...], Dict]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        if isinstance(inputs, Mapping):
            return self._prepare_input(inputs)
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.training_args.device))
            else:
                prepared.append(self._prepare_input(x))
        return prepared

    def get_train_dataloader(self, train_dataset, use_small_batch_size=False) -> DataLoader:
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.training_args.world_size > 1:
            seed = self.training_args.data_seed if self.training_args.data_seed is not None else self.training_args.seed
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.training_args.world_size,
                rank=self.training_args.process_index,
                seed=seed,
            )
        else:
            train_sampler = RandomSampler(train_dataset)

        if use_small_batch_size:
            train_batch_size = 128 // self.training_args.world_size
        else:
            train_batch_size = self.training_args.train_batch_size

        return DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )

    def compute_loss(self, inputs):
        query, passage = inputs
        return self.model(query=query, passage=passage).loss

    def train_step(self, batch):
        if self.use_amp:
            if version.parse(torch.__version__) > version.parse("1.7.1"):
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    kl_loss = self.compute_loss(batch)
            else:
                with torch.cuda.amp.autocast():
                    kl_loss = self.compute_loss(batch)
        else:
            kl_loss = self.compute_loss(batch)

        return kl_loss

    def train(self):
        global_step = 0
        prev_global_step = 0
        if self.training_args.max_steps > 0:
            dataset_epochs = [0] * len(self.train_dataloader)
        for epoch in range(self.num_train_epochs):
            if self.training_args.multi_task:
                for dataloader in self.train_dataloader:
                    if isinstance(dataloader.sampler, DistributedSampler):
                        dataloader.sampler.set_epoch(epoch)
            else:
                if isinstance(self.train_dataloader.sampler, DistributedSampler):
                    self.train_dataloader.sampler.set_epoch(epoch)
            self.epoch = copy.deepcopy(epoch)
            self.model.train()
            losses = AverageMeter('Loss', ':.4')
            progress = ProgressMeter(
                self.max_step if self.training_args.max_steps > 0 else self.num_training_steps,
                [losses],
                prefix="Epoch: [{}]".format(epoch))
            step = 0

            num_training_steps = [len(dataloader) for dataloader in self.train_dataloader]
            data_src_indices = []
            iterators = []
            for source, src_its in enumerate(num_training_steps):
                if self.training_args.max_steps > 0:
                    src_its = self.training_args.max_steps * self.training_args.gradient_accumulation_steps
                data_src_indices.extend([source] * src_its)
                train_dataloader = self.train_dataloader[source]
                iterators.append(iter(train_dataloader))

            epoch_rnd = random.Random(self.training_args.seed + epoch)
            epoch_rnd.shuffle(data_src_indices)

            for i, source_idx in enumerate(data_src_indices):
                try:
                    it = iterators[source_idx]
                    batch = next(it)
                except:
                    if self.training_args.max_steps > 0:
                        dataset_epochs[source_idx] += 1
                        dataloader = self.train_dataloader[source_idx]
                        if isinstance(dataloader.sampler, DistributedSampler):
                            dataloader.sampler.set_epoch(dataset_epochs[source_idx])
                    iterators[source_idx] = iter(self.train_dataloader[source_idx])
                    it = iterators[source_idx]
                    batch = next(it)
                batch = self._prepare_inputs(batch)
                kl_loss = self.train_step(batch)

                # gradient accumulation steps
                if not self.training_args.distributed_contrast:
                    if self.use_amp:
                        self.scaler.scale(kl_loss).backward()
                    else:
                        kl_loss.backward()

                if (step + 1) % self.training_args.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_args.max_grad_norm)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    global_step += 1

                if self.training_args.negatives_x_device and not self.training_args.distributed_contrast:
                    kl_loss = kl_loss / self._dist_loss_scale_factor
                    loss_list = [torch.zeros_like(kl_loss) for _ in range(dist.get_world_size())]
                    dist.all_gather(tensor_list=loss_list, tensor=kl_loss.contiguous())
                    loss = torch.mean(torch.stack(loss_list, dim=0), dim=0)
                    losses.update(loss.item())
                else:
                    losses.update(kl_loss.item())

                step += 1
                if self.training_args.max_steps > 0 and self.training_args.gradient_accumulation_steps > 1:
                    if global_step != 0 and global_step != prev_global_step \
                            and global_step % self.training_args.print_steps == 0 and \
                            self.training_args.process_index in [-1, 0]:
                        progress.display(global_step)
                        prev_global_step = global_step
                else:
                    if step % (self.training_args.print_steps * self.training_args.gradient_accumulation_steps) == 0 \
                            and self.training_args.process_index in [-1, 0]:
                        progress.display(step // self.training_args.gradient_accumulation_steps)

                if global_step != 0 and global_step % self.training_args.save_steps == 0 and \
                        self.training_args.process_index in [-1, 0]:
                    checkpoint_folder = f"checkpoint-{global_step}"
                    output_dir = os.path.join(self.training_args.output_dir, checkpoint_folder)
                    self._save(self.model, output_dir)
                if global_step >= self.max_step:
                    break
            if global_step >= self.max_step:
                break
