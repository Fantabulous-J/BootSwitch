from dataclasses import dataclass, field
from typing import Optional, List

import torch
from transformers import TrainingArguments
from transformers.utils import cached_property, logging

from utils import _infer_slurm_init

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    query_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    passage_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    shared_encoder: bool = field(
        default=True,
        metadata={"help": "weight sharing between qry passage encoders"}
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    corpus_file: str = field(default="corpus.tsv", metadata={"help": "corpus text path"})
    query_file: str = field(default="train.query.txt", metadata={"help": "query text path"})
    qrels_file: str = field(default="qrels.train.tsv", metadata={"help": "query passage relation path"})

    train_n_passages: int = field(default=8)
    no_shuffle_positive: bool = field(default=False)
    sample_hard_negative_prob: float = field(default=1.0)

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=10)
    encode_shard_index: int = field(default=0)
    split: str = field(default='train', metadata={"help": "dataset splits"})

    max_query_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_passage_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_query_passage_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage & query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
    )


@dataclass
class BiEncoderTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})

    print_steps: int = field(default=100, metadata={"help": "step for displaying"})

    distributed_port: int = field(default=None, metadata={"help": "port for multi-node multi-gpu distributed training "
                                                                  "using slurm"})
    multi_task: bool = field(default=False, metadata={"help": "use multi-task training"})
    noisy: bool = field(default=False, metadata={"help": "add noise to query and passage"})
    noise_type: str = field(default='shuffle', metadata={"help": "method to add noise on query and passage"})
    noise_prob: float = field(default=0.1)

    @cached_property
    def _setup_devices(self) -> "torch.device":
        if self.distributed_port:
            logger.info("PyTorch: setting up devices")
            distributed_init_method, local_rank, world_size, device_id = _infer_slurm_init(self.distributed_port)
            self.local_rank = local_rank
            torch.distributed.init_process_group(
                backend="nccl", init_method=distributed_init_method, world_size=world_size, rank=local_rank
            )

            logger.info("local rank {}, device id {}".format(local_rank, device_id))
            self._n_gpu = 1
            if device_id is None:
                device = torch.device("cuda")
            else:
                device = torch.device("cuda", device_id)
                if device.type == "cuda":
                    torch.cuda.set_device(device)

            return device
        else:
            return super(BiEncoderTrainingArguments, self)._setup_devices
