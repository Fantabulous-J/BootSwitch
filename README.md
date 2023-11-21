# Boot and Switch: Alternating Distillation for Zero-Shot Dense Retrieval
Source code for our EMNLP 2023 Findings Paper "Boot and Switch: Alternating Distillation for Zero-Shot Dense Retrieval".

## Install environment
```shell
pip install -r requirements.txt
```

## Evaluation
### Models
- [fanjiang98/ABEL-Query-Encoder-Warmup](https://huggingface.co/fanjiang98/ABEL-Query-Encoder-Warmup): warm-up query encoder.
- [fanjiang98/ABEL-Passage-Encoder-Warmup](https://huggingface.co/fanjiang98/ABEL-Passage-Encoder-Warmup): warm-up passage encoder.
- [fanjiang98/ABEL-Query-Encoder](https://huggingface.co/fanjiang98/ABEL-Query-Encoder): query encoder.
- [fanjiang98/ABEL-Passage-Encoder](https://huggingface.co/fanjiang98/ABEL-Passage-Encoder): passage encoder.
### BEIR
#### Download Dataset
```shell
mkdir -p beir
cd beir
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
unzip scifact.zip
cd ../
```
Other datasets can be downloaded from [here](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/). Note that BioASQ, TREC-NEWS, Robust04 and Signal-1M are not publicly available, please refer to [here](https://github.com/beir-cellar/beir/wiki/Datasets-available) for more details.

#### Retrieval
```shell
python eval_beir.py \
    --data_dir beir \ 
    --task scifact \
    --query_encoder_path fanjiang98/ABEL-Query-Encoder \
    --passage_encoder_path fanjiang98/ABEL-Passage-Encoder \
    --max_seq_length 512 \
    --pooling_mode mean
```
### Cross-task Cross-domain dataset
#### Download Dataset
```shell
wget https://homes.cs.washington.edu/~akari/tart/cross_task_cross_domain_final.zip
unzip cross_task_cross_domain_final.zip
```
#### Retrieval
```shell
python eval_x2.py \
    --data_dir corss_task_cross_domain_final \
    --task ambig \
    --query_encoder_path fanjiang98/ABEL-Query-Encoder \
    --passage_encoder_path fanjiang98/ABEL-Passage-Encoder
```

### Open-Domain Question Answering
#### Download Dataset
```shell
mkdir -p ODQA
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gunzip -d psgs_w100.tsv.gz
```

#### Generate Embeddings
Encode Query
```shell
MODEL_PATH=fanjiang98/ABEL-Query-Encoder
DATA_PATH=ODQA

mkdir -p ${DATA_PATH}/encoding
python encode.py \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${MODEL_PATH} \
    --train_dir ${DATA_PATH} \
    --fp16 \
    --per_device_eval_batch_size 2048 \
    --encode_is_qry \
    --shared_encoder False \
    --max_query_length 128 \
    --query_file nq-test.qa.csv \
    --dataloader_num_workers 4 \
    --encoded_save_path ${DATA_PATH}/encoding/nq_test_query_embedding.pt
```
Encode Corpus
```shell
MODEL_PATH=fanjiang98/ABEL-Passage-Encoder
DATA_PATH=ODQA

mkdir -p ${DATA_PATH}/encoding
for i in $(seq -f "%02g" 0 19)
do
python encode.py \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${MODEL_PATH} \
    --train_dir ${DATA_PATH} \
    --fp16 \
    --corpus_file psgs_w100.tsv \
    --shared_encoder False \
    --max_passage_length 512 \
    --per_device_eval_batch_size 128 \
    --encode_shard_index $i \
    --encode_num_shard 10 \
    --dataloader_num_workers 4 \
    --encoded_save_path ${DATA_PATH}/encoding/embedding_split${i}.pt
done
```
#### Retrieve
```shell
DATA_PATH=ODQA
python retriever.py \
    --query_embeddings ${DATA_PATH}/encoding/nq_test_query_embedding.pt \
    --passage_embeddings ${DATA_PATH}/encoding/'embedding_split*.pt' \
    --depth 100 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to ${DATA_PATH}/nq.rank.txt
```
We use pyserini for evaluation:
```shell
python convert_result_to_trec.py --input ${DATA_PATH}/nq.rank.txt --output ${DATA_PATH}/nq.rank.txt.trec

python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
    --topics dpr-nq-test \
    --index wikipedia-dpr \
    --input nq.rank.txt.trec \
    --output run.nq.test.json
python -m pyserini.eval.evaluate_dpr_retrieval 
    --retrieval run.nq.test.json  \
    --topk 1 5 20 100
```

### Training
Please download the training data from [OneDrive](https://unimelbcloud-my.sharepoint.com/:f:/g/personal/jifj_student_unimelb_edu_au/El06TA1UpUlJpa2VN7VlvYkBVz-YbRv3-2SyMSBQwXfrTQ?e=bPav3Q) and put them on corresponding directories under `beir`.

Tokenizer query and corpus
```shell
python tokenize_query_passage.py
```
We use slurm for training on 8 80G A100:
```shell
bash scripts/beir/train_biencoder_sent_crop_beir.sh
```
