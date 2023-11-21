import json
import logging
import multiprocessing as mp
import os

from beir import LoggingHandler
from tqdm import tqdm
from transformers import AutoTokenizer, T5TokenizerFast

from dataloader import GenericDataLoader

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


def encode(text, max_length):
    return tokenizer.encode_plus(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )


def process_query_line(item):
    qid, query = item

    encoded_query = encode(query, max_length=128).data

    return {
        'qid': qid,
        'text': encoded_query
    }


def process_passage_line(item):
    pid, passage = item
    title = passage['title']
    text = passage['text']
    passage = (title + " " + text).strip()
    encoded_passage = encode(passage, max_length=256).data

    return {
        'pid': pid,
        'text': encoded_passage
    }


if __name__ == '__main__':
    directory = 'beir'
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever', use_fast=True)

    for task in ['fiqa', 'scifact', 'arguana', 'climate-fever', 'dbpedia-entity', 'cqadupstack', 'quora', 'scidocs',
                 'nfcorpus', 'signal1m', 'trec-covid', 'webis-touche2020', 'hotpotqa', 'nq', 'robust04', 'trec-news',
                 'bioasq']:
        logger.info(f'### {task} ###')

        if task == 'cqadupstack':
            corpus = []
            for subtask in ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats',
                            'tex', 'unix', 'webmasters', 'wordpress']:
                with open('{}/{}/{}/corpus.jsonl'.format(directory, task, subtask)) as f:
                    subtask_corpus = [json.loads(jsonline) for jsonline in f.readlines()]
                    for doc in subtask_corpus:
                        doc['_id'] = "{}_{}".format(subtask, doc['_id'])
                    corpus.extend(subtask_corpus)
        else:
            with open('{}/{}/corpus.jsonl'.format(directory, task)) as f:
                corpus = [json.loads(jsonline) for jsonline in f.readlines()]

        pid = 0
        jsonl = []
        for doc in tqdm(corpus):
            if len(doc['text'].strip()) == 0 or len(doc['_id']) == 0:
                continue
            doc['text'] = ' '.join(doc['text'].split('\n'))
            doc['text'] = ' '.join(doc['text'].split('\r'))
            doc['text'] = ' '.join(doc['text'].split('\t'))

            doc['title'] = ' '.join(doc['title'].split('\n'))
            doc['title'] = ' '.join(doc['title'].split('\r'))
            doc['title'] = ' '.join(doc['title'].split('\t'))

            jsonl.append({
                "id": pid,
                "contents": doc['text']
            })
            pid += 1

        print(f'write {pid} passages into {directory}/{task}/corpus.json')
        with open('{}/{}/corpus.json'.format(directory, task), 'w') as f:
            json.dump(jsonl, f)

        corpus = GenericDataLoader(os.path.join(directory, task), corpus_file='corpus.json').load_corpus()
        corpus = [(pid, passage) for pid, passage in corpus.items()]
        queries = GenericDataLoader(os.path.join(directory, task), query_file='sent_query.txt').load_queries()
        queries = [(qid, query) for qid, query in queries.items()]

        output_path = '{}/{}/tokenized_query.jsonl'.format(directory, task)
        with open(output_path, 'w') as f:
            with mp.Pool(processes=4) as pool:
                for jsonl in tqdm(pool.imap(process_query_line, queries, chunksize=10000)):
                    if isinstance(tokenizer, T5TokenizerFast):
                        jsonl['text']['input_ids'] = jsonl['text']['input_ids'][:-1]
                    f.write(json.dumps(jsonl) + '\n')

        output_path = '{}/{}/tokenized_passage.jsonl'.format(directory, task)
        with open(output_path, 'w') as f:
            with mp.Pool(processes=32) as pool:
                for jsonl in tqdm(pool.imap(process_passage_line, corpus, chunksize=10000)):
                    if isinstance(tokenizer, T5TokenizerFast):
                        jsonl['text']['input_ids'] = jsonl['text']['input_ids'][:-1]
                    f.write(json.dumps(jsonl) + '\n')
