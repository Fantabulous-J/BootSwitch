import json
import logging
import os
import pickle
import random
import subprocess
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def _infer_slurm_init(distributed_port) -> Tuple[str, int, int, int]:
    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    logger.info("SLURM_JOB_NODELIST: %s", node_list)

    if node_list is None:
        raise RuntimeError("Can't find SLURM node_list from env parameters")

    local_rank = None
    world_size = None
    distributed_init_method = None
    device_id = None
    try:
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
        distributed_init_method = "tcp://{host}:{port}".format(
            host=hostnames.split()[0].decode("utf-8"),
            port=distributed_port,
        )
        nnodes = int(os.environ.get("SLURM_NNODES"))
        logger.info("SLURM_NNODES: %s", nnodes)
        ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
        if ntasks_per_node is not None:
            ntasks_per_node = int(ntasks_per_node)
            logger.info("SLURM_NTASKS_PER_NODE: %s", ntasks_per_node)
            if ntasks_per_node == 1:
                gpus_per_node = torch.cuda.device_count()
                node_id = int(os.environ.get("SLURM_NODEID"))
                local_rank = node_id * gpus_per_node
                device_id = 0
                world_size = nnodes * gpus_per_node
                logger.info("node_id: %s", node_id)
            else:
                world_size = ntasks_per_node * nnodes
                proc_id = os.environ.get("SLURM_PROCID")
                local_id = os.environ.get("SLURM_LOCALID")
                logger.info("SLURM_PROCID %s", proc_id)
                logger.info("SLURM_LOCALID %s", local_id)
                local_rank = int(proc_id)
                device_id = int(local_id)
        else:
            ntasks = int(os.environ.get("SLURM_NTASKS"))
            logger.info("SLURM_NTASKS: %s", ntasks)
            world_size = ntasks
            proc_id = os.environ.get("SLURM_PROCID")
            local_id = os.environ.get("SLURM_LOCALID")
            logger.info("SLURM_PROCID %s, SLURM_LOCALID %s", proc_id, local_id)
            local_rank = int(proc_id)

    except subprocess.CalledProcessError as e:  # scontrol failed
        raise e
    except FileNotFoundError:  # Slurm is not installed
        pass
    return distributed_init_method, local_rank, world_size, device_id


def shuffle_word(x, p=0.1, **kwargs):
    count = (np.random.rand(len(x)) < p).sum()
    """Shuffles any n number of values in a list"""
    indices_to_shuffle = random.sample(range(len(x)), k=count)
    to_shuffle = [x[i] for i in indices_to_shuffle]
    random.shuffle(to_shuffle)
    for index, value in enumerate(to_shuffle):
        old_index = indices_to_shuffle[index]
        x[old_index] = value
    return x


def delete_word(x, p=0.1, **kwargs):
    mask = np.random.rand(len(x))
    x = [e for e, m in zip(x, mask) if m > p]
    return x


def replace_word(x, min_random, max_random, p=0.1, **kwargs):
    mask = np.random.rand(len(x))
    x = [e if m > p else random.randint(min_random, max_random) for e, m in zip(x, mask)]
    return x


def mask_word(x, mask_id=103, p=0.1):
    mask = np.random.rand(len(x))
    x = [e if m > p else mask_id for e, m in zip(x, mask)]
    return x


def noise_sequentially(x, mask_id=103, p=0.1):
    x = shuffle_word(x, p=p)
    x = delete_word(x, p=p)
    x = mask_word(x, mask_id=mask_id, p=p)
    return x


noise_strategy = {
    'shuffle': shuffle_word,
    'delete': delete_word,
    'replace': replace_word,
    'mask': mask_word,
    'sequential': noise_sequentially
}