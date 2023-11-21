import copy
import csv
import mmap
import re

import pandas as pd
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Union, List

import torch
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, T5TokenizerFast, T5Tokenizer

from arguments import DataArguments
from utils import noise_strategy

logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class MemoryMappedDataset(Dataset):
    def __init__(self, path):
        self.file = open(path, mode="r")
        self.mm = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
        self.offset_dict = {0: self.mm.tell()}
        line = self.mm.readline()
        self.count = 0
        while line:
            self.count += 1
            offset = self.mm.tell()
            self.offset_dict[self.count] = offset
            line = self.mm.readline()

    def __len__(self):
        return self.count

    def process_line(self, line):
        return line

    def __getitem__(self, index):
        offset = self.offset_dict[index]
        self.mm.seek(offset)
        line = self.mm.readline()
        return self.process_line(line)


class JsonlDataset(MemoryMappedDataset):
    def __init__(self, path):
        super(JsonlDataset, self).__init__(path)

    def __getitem__(self, index):
        try:
            return super(JsonlDataset, self).__getitem__(eval(index))
        except:
            return super(JsonlDataset, self).__getitem__(index)

    def process_line(self, line):
        return json.loads(line)


class QueryDataset(MemoryMappedDataset):
    def __init__(self, path):
        super(QueryDataset, self).__init__(path)

    def __getitem__(self, index):
        return super(QueryDataset, self).__getitem__(eval(index))

    def process_line(self, line):
        query_id, text = line.decode().strip().split('\t')
        return text


class PassageDataset(MemoryMappedDataset):
    def __init__(self, path):
        super(PassageDataset, self).__init__(path)

    def __getitem__(self, index):
        return super(PassageDataset, self).__getitem__(eval(index))

    def process_line(self, line):
        try:
            corpus_id, title, text = line.decode().strip().split('\t')
        except:
            corpus_id, text = line.decode().strip().split('\t')
            title = ""
        return {
            'text': text,
            'title': "" if title == '-' else title,
        }


class PassageJsonDataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as json_file:
            with mmap.mmap(json_file.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_object:
                self.data = json.load(mmap_object)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[eval(index)]
        corpus_id, text = str(line['id']), line['contents']
        return {
            'text': text,
            'title': ""
        }


class GenericDataLoader:

    def __init__(self, data_folder: str = None, corpus_file: str = "corpus.tsv", query_file: str = "train.query.txt",
                 qrel_file: str = "train-hard-negatives.jsonl", use_mmap: bool = False):
        self.corpus = {}
        self.queries = {}
        self.qrels = []

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file

        if ',' in query_file:
            query_file = query_file.split(',')
            self.query_file = [os.path.join(data_folder, file) for file in query_file]
        else:
            self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file

        if ',' in qrel_file:
            qrel_file = qrel_file.split(',')
            if data_folder and 'nq' not in qrel_file:
                self.qrel_file = [os.path.join(data_folder, file) for file in qrel_file]
            else:
                self.qrel_file = qrel_file
        else:
            self.qrel_file = os.path.join(data_folder, qrel_file) if data_folder and 'nq' not in qrel_file \
                else qrel_file

        self.use_mmap = use_mmap

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load(self, split="test"):

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s",
                        list(self.corpus.values())[0] if isinstance(self.corpus, dict) else self.corpus['0'])

        if not len(self.queries):
            self.load_queries()

        if not len(self.qrels):
            self.load_qrels()

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> Dict[str, Dict[str, str]]:

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s",
                        list(self.corpus.values())[0] if isinstance(self.corpus, dict) else self.corpus['0'])

        return self.corpus

    def load_queries(self) -> Dict[str, str]:
        logger.info("Loading Queries...")
        if isinstance(self.query_file, List):
            self.queries = []
            for query_file in self.query_file:
                queries = self._load_queries(query_file)
                logger.info("Loaded %d Queries.", len(queries))
                logger.info("Query Example: %s",
                            list(queries.values())[0] if not self.use_mmap or 'dev' in query_file
                                                         or 'dl' in query_file else queries['0'])
                self.queries.append(queries)
        else:
            self.queries = self._load_queries(self.query_file)
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s",
                        list(self.queries.values())[0] if not self.use_mmap or 'dev' in self.query_file
                                                          or 'dl' in self.query_file else self.queries['0'])

        return self.queries

    def load_qrels(self):
        def _load_qrels(qrel_file):
            if self.use_mmap:
                qrels = JsonlDataset(qrel_file)
            else:
                qrels = []
                with open(qrel_file, 'r') as f:
                    for jsonline in tqdm(f.readlines()):
                        if 'sent' in qrel_file:
                            qrels.append(jsonline)
                        else:
                            example = json.loads(jsonline)
                            qrels.append(example)
            return qrels

        if isinstance(self.qrel_file, List):
            for qrel_file in self.qrel_file:
                qrels = _load_qrels(qrel_file)
                logger.info("Loaded %d Queries from %s.", len(qrels), qrel_file)
                self.qrels.append(qrels)
        else:
            self.qrels = _load_qrels(self.qrel_file)
            logger.info("Loaded %d Queries from %s.", len(self.qrels), self.qrel_file)

    def _load_corpus(self):
        if self.corpus_file.endswith('json'):
            if self.use_mmap:
                self.corpus = PassageJsonDataset(self.corpus_file)
            else:
                with open(self.corpus_file, encoding='utf-8') as fIn:
                    for line in tqdm(json.load(fIn)):
                        corpus_id, text = str(line['id']), line['contents']
                        self.corpus[corpus_id] = {
                            'text': text,
                            'title': ""
                        }
        elif self.corpus_file.endswith('jsonl'):
            if self.use_mmap:
                self.corpus = JsonlDataset(self.corpus_file)
            else:
                with open(self.corpus_file, encoding='utf-8') as fIn:
                    for jsonline in fIn.readlines():
                        example = json.loads(jsonline)
                        self.corpus[example['pid']] = example
        else:
            self.check(fIn=self.corpus_file, ext="tsv")
            if self.use_mmap:
                self.corpus = PassageDataset(self.corpus_file)
            else:
                with open(self.corpus_file, encoding='utf-8') as fIn:
                    reader = csv.reader(fIn, delimiter="\t")
                    for row in tqdm(reader):
                        if not row[0] == "id":
                            self.corpus[row[0]] = {
                                "title": row[2],
                                "text": row[1]
                            }

    def _load_queries(self, query_file):
        queries = {}
        if query_file.endswith('jsonl'):
            if self.use_mmap:
                if 'dev' in query_file or 'dl' in query_file:
                    with open(query_file, encoding='utf-8') as fIn:
                        for qid, jsonline in enumerate(fIn.readlines()):
                            example = json.loads(jsonline)
                            queries[example['qid']] = example
                else:
                    queries = JsonlDataset(query_file)
            else:
                with open(query_file, encoding='utf-8') as fIn:
                    queries = {str(qid): json.loads(jsonline) for qid, jsonline in enumerate(fIn.readlines())}
        else:
            if self.use_mmap:
                queries = QueryDataset(query_file)
            else:
                if query_file.endswith('tsv'):
                    with open(query_file, encoding='utf-8') as fIn:
                        reader = csv.reader(fIn, delimiter="\t")
                        for row in tqdm(reader):
                            if not row[0] == "qid":
                                queries[str(len(queries))] = row[1]
                else:
                    with open(query_file, encoding='utf-8') as fIn:
                        for line in fIn:
                            query_id, text = line.strip().split('\t')
                            queries[query_id] = text

        return queries


class BiEncoderDataset(Dataset):
    def __init__(self,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 qrels: Union[Dataset, List[Dataset], List[Dict], List[List[Dict]]],
                 tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments,
                 sep: str = " ",
                 pretokenized: bool = False,
                 trainer=None):
        super(BiEncoderDataset, self).__init__()

        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.sep = sep
        self.pretokenized = pretokenized
        self.trainer = trainer

    def __len__(self):
        if isinstance(self.qrels[0], Dataset) or isinstance(self.qrels[0], List):
            return min(len(qrels) for qrels in self.qrels)
        return len(self.qrels)

    def __getitem__(self, idx):
        if isinstance(self.qrels[0], Dataset) or isinstance(self.qrels[0], List):
            dataset_idx = random.choice(range(len(self.qrels)))
            example = self.qrels[dataset_idx][idx]
            assert isinstance(self.queries, List), type(self.queries)
            queries = self.queries[dataset_idx]
        else:
            example = self.qrels[idx]
            queries = self.queries
        if isinstance(example, str):
            example = json.loads(example)
        _hashed_seed = hash(idx + self.trainer.args.seed)
        epoch = int(self.trainer.state.epoch if hasattr(self.trainer, 'state') else self.trainer.epoch)

        qid = example['qid']
        if self.pretokenized:
            encoded_query = queries[qid]['text']
            if isinstance(self.tokenizer, T5TokenizerFast) or isinstance(self.tokenizer, T5Tokenizer):
                encoded_query['input_ids'] = encoded_query['input_ids'] + [self.tokenizer.eos_token_id]
        else:
            encoded_query = self.encode(queries[qid], max_length=self.data_args.max_query_length)

        encoded_passages = []

        if 'pos' in example:
            pos_pids = example['pos']
        else:
            assert 'pids' in example
            pos_pids = example['pids'][:10]
        if self.data_args.no_shuffle_positive:
            pos_pid = pos_pids[0]
        else:
            pos_pid = pos_pids[(_hashed_seed + epoch) % len(pos_pids)]

        if 'neg' in example:
            neg_pids = example['neg']
        else:
            assert 'pids' in example
            neg_pids = example['pids'][-5:]
        negative_size = self.data_args.train_n_passages - 1
        if len(neg_pids) < negative_size:
            if len(neg_pids) == 0:
                assert len(example['random_neg']) >= negative_size, len(example['random_neg'])
                negs = random.sample(example['random_neg'], negative_size)
            else:
                negs = random.choices(neg_pids, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        else:
            if random.random() <= self.data_args.sample_hard_negative_prob:
                _offset = epoch * negative_size % len(neg_pids)
                negs = [x for x in neg_pids]
                random.Random(_hashed_seed).shuffle(negs)
                negs = negs * 2
                negs = negs[_offset: _offset + negative_size]
            else:
                assert len(example['random_neg']) >= negative_size, len(example['random_neg'])
                negs = random.sample(example['random_neg'], negative_size)

        for pid in [pos_pid] + negs:
            if self.pretokenized:
                encoded_passage = self.corpus[pid]['text']
                if isinstance(self.tokenizer, T5TokenizerFast) or isinstance(self.tokenizer, T5Tokenizer):
                    encoded_passage['input_ids'] = encoded_passage['input_ids'] + [self.tokenizer.eos_token_id]
            else:
                passage = self.corpus[pid]
                title = passage['title']
                text = passage['text']
                passage = (title + self.sep + text).strip()
                encoded_passage = self.encode(passage, max_length=self.data_args.max_passage_length)
            encoded_passages.append(encoded_passage)

        return encoded_query, encoded_passages

    def encode(self, text, max_length):
        return self.tokenizer.encode_plus(
            text,
            max_length=max_length,
            truncation='only_first',
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )


class NoisyBiEncoderDataset(BiEncoderDataset):
    def __init__(self,
                 corpus: Dict[str, Dict[str, str]],
                 queries: Dict[str, str],
                 qrels: Union[Dataset, List[Dict]],
                 tokenizer: PreTrainedTokenizer,
                 data_args: DataArguments,
                 sep: str = " ",
                 pretokenized: bool = False,
                 trainer=None,
                 noise_type: str = 'sequential',
                 noise_prob: float = 0.1):
        super(NoisyBiEncoderDataset, self).__init__(corpus=corpus,
                                                    queries=queries,
                                                    qrels=qrels,
                                                    tokenizer=tokenizer,
                                                    data_args=data_args,
                                                    sep=sep,
                                                    pretokenized=pretokenized,
                                                    trainer=trainer)
        assert noise_type in noise_strategy.keys(), noise_type
        self.noise_strategy = noise_strategy[noise_type]
        self.noise_prob = noise_prob

    def __getitem__(self, idx):
        encoded_query, encoded_passages = super(NoisyBiEncoderDataset, self).__getitem__(idx)

        encoded_noisy_query = copy.deepcopy(encoded_query)
        if isinstance(self.tokenizer, T5TokenizerFast) or isinstance(self.tokenizer, T5Tokenizer):
            input_ids = self.noise_strategy(encoded_noisy_query['input_ids'][:-1], p=self.noise_prob)
            encoded_noisy_query['input_ids'] = input_ids + [self.tokenizer.eos_token_id]
        else:
            input_ids = self.noise_strategy(encoded_noisy_query['input_ids'][1:-1], p=self.noise_prob)
            encoded_noisy_query['input_ids'] = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]

        encoded_noisy_passages = copy.deepcopy(encoded_passages)
        for passage in encoded_noisy_passages:
            if isinstance(self.tokenizer, T5TokenizerFast) or isinstance(self.tokenizer, T5Tokenizer):
                input_ids = self.noise_strategy(passage['input_ids'][:-1], p=self.noise_prob)
                passage['input_ids'] = input_ids + [self.tokenizer.eos_token_id]
            else:
                input_ids = self.noise_strategy(passage['input_ids'][1:-1], p=self.noise_prob)
                passage['input_ids'] = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]

        return encoded_noisy_query, encoded_noisy_passages


class EncodeDataset(Dataset):
    def __init__(self,
                 data: Dict,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int,
                 is_query: bool,
                 start: int,
                 end: int,
                 sep: str = " ",
                 pretokenized: bool = False):
        super(EncodeDataset, self).__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_query = is_query

        if pretokenized and not isinstance(data, dict):
            self.data = data
        else:
            self.data = []
            for idx, text in data.items():
                self.data.append((idx, text))
            self.data = self.data[start:end]

        self.start = start
        self.end = end
        self.sep = sep
        self.pretokenized = pretokenized

    def __len__(self):
        if self.pretokenized:
            return self.end - self.start
        return len(self.data)

    def __getitem__(self, idx):
        if self.pretokenized:
            data = self.data[self.start + idx]
            if isinstance(data, tuple):
                _, data = data
            text_id = data['qid'] if self.is_query else data['pid']
            encoded_text = data['text']
        else:
            text_id, text = self.data[idx]
            if not self.is_query:
                text = (text['title'] + self.sep + text['text']).strip()

            encoded_text = self.tokenizer.encode_plus(
                text,
                max_length=self.max_length,
                truncation='only_first',
                padding=False,
                return_token_type_ids=False,
            )
        return text_id, encoded_text


@dataclass
class BiEncoderCollator(DataCollatorWithPadding):
    max_query_length: int = 256
    max_passage_length: int = 256

    def _encode(self, features):
        batch_query = [f[0] for f in features]
        batch_passage = [f[1] for f in features]
        if isinstance(batch_query[0], list):
            batch_query = sum(batch_query, [])
        if isinstance(batch_passage[0], list):
            batch_passage = sum(batch_passage, [])

        batch_query = self.tokenizer.pad(
            batch_query,
            padding=True,
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        batch_passage = self.tokenizer.pad(
            batch_passage,
            padding=True,
            max_length=self.max_passage_length,
            return_tensors="pt",
        )

        if len(features[0]) == 3:
            batch_augment_passage = [f[2] for f in features]
            if isinstance(batch_augment_passage[0], list):
                batch_augment_passage = sum(batch_augment_passage, [])

            batch_augment_passage = self.tokenizer.pad(
                batch_augment_passage,
                padding=True,
                max_length=self.max_passage_length,
                return_tensors="pt"
            )
            return batch_query, batch_passage, batch_augment_passage

        if len(features[0]) == 4:
            batch_teacher_query = [f[2] for f in features]
            batch_teacher_passage = [f[3] for f in features]
            if isinstance(batch_teacher_query[0], list):
                batch_teacher_query = sum(batch_teacher_query, [])
            if isinstance(batch_teacher_passage[0], list):
                batch_teacher_passage = sum(batch_teacher_passage, [])

            batch_teacher_query = self.tokenizer.pad(
                batch_teacher_query,
                padding=True,
                max_length=self.max_query_length,
                return_tensors="pt"
            )
            batch_teacher_passage = self.tokenizer.pad(
                batch_teacher_passage,
                padding=True,
                max_length=self.max_passage_length,
                return_tensors="pt"
            )
            return batch_query, batch_passage, batch_teacher_query, batch_teacher_passage

        return batch_query, batch_passage

    def __call__(self, features):
        return self._encode(features)


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch_text_id = [x[0] for x in features]
        batch_text = [x[1] for x in features]
        batch_text = super().__call__(batch_text)

        return batch_text_id, batch_text
