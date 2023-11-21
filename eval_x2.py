import argparse
import logging
from typing import List, Dict, Union

import numpy as np
import torch.multiprocessing as mp
from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from torch import Tensor

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class SentenceBERT:
    def __init__(self, q_model, doc_model, sep: str = " "):
        self.sep = sep

        self.q_model = q_model
        self.doc_model = doc_model

    def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for process_id, device_name in enumerate(target_devices):
            p = ctx.Process(target=SentenceTransformer._encode_multi_process_worker,
                            args=(process_id, device_name, self.doc_model, input_queue, output_queue), daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    def stop_multi_process_pool(self, pool: Dict[str, object]):
        output_queue = pool['output']
        [output_queue.get() for _ in range(len(pool['processes']))]
        return self.doc_model.stop_multi_process_pool(pool)

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[
        List[Tensor], np.ndarray, Tensor]:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dict[str, List]], batch_size: int = 8, **kwargs) -> \
            Union[List[Tensor], np.ndarray, Tensor]:
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][
                    i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for
                         doc in corpus]
        return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs)

    # Encoding corpus in parallel
    def encode_corpus_parallel(self, corpus: Union[List[Dict[str, str]], Dataset], pool: Dict[str, str],
                               batch_size: int = 8, chunk_id: int = None, **kwargs):
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][
                    i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for
                         doc in corpus]

        if chunk_id is not None and chunk_id >= len(pool['processes']):
            output_queue = pool['output']
            output_queue.get()

        input_queue = pool['input']
        input_queue.put([chunk_id, batch_size, sentences])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--query_encoder_path', type=str, required=True)
    parser.add_argument('--passage_encoder_path', type=str, required=True)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--pooling_mode', type=str, default='mean')
    args = parser.parse_args()

    data_dir = args.data_dir
    task = args.task
    query_encoder_path = args.query_encoder_path
    passage_encoder_path = args.passage_encoder_path
    max_seq_length = args.max_seq_length
    pooling_mode = args.pooling_mode

    assert task in ['ambig', 'wikiqa', 'gooaq_technical', 'linkso_py', 'codesearch_py'], \
        "task must be one of ambig, wikiqa, gooaq_technical, linkso_py, codesearch_py"

    logging.info("###### {} ######".format(task))

    logger.info("Load query encoder from {}, max_seq_length {}".format(query_encoder_path, max_seq_length))
    bert = models.Transformer(query_encoder_path, max_seq_length=max_seq_length)
    pool = models.Pooling(bert.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    query_model = SentenceTransformer(modules=[bert, pool])

    logger.info("Load passage encoder from {}, max_seq_length {}".format(passage_encoder_path, max_seq_length))
    bert = models.Transformer(passage_encoder_path, max_seq_length=max_seq_length)
    pool = models.Pooling(bert.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    passage_model = SentenceTransformer(modules=[bert, pool])

    model = DRES(SentenceBERT(query_model, passage_model, sep=' '), batch_size=16)

    retriever = EvaluateRetrieval(model, score_function="dot")

    corpus, queries, qrels = GenericDataLoader(data_folder='{}/{}'.format(data_dir, task)).load(split="test")

    for k in corpus.keys():
        if type(corpus[k]["text"]) is float:
            corpus[k]["text"] = corpus[k]["title"]
        if type(corpus[k]["title"]) is float:
            corpus[k]["title"] = ""

    results = retriever.retrieve(corpus, queries)

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    logging.info('\n')
