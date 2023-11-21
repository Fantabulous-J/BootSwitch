#!/bin/bash
#
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=480G
#SBATCH --partition=gpu-a100
#SBATCH -A punim2015
#SBATCH --gres=gpu:A100:4

source /home/fanjiang/miniconda3/bin/activate DPR

srun python train_crop_sent.py --distributed_port 23333 \
  --output_dir ./checkpoints/ABEL \
  --model_name_or_path facebook/contriever \
  --query_model_name_or_path fanjiang98/ABEL-Query-Encoder-Warmup \
  --passage_model_name_or_path fanjiang98/ABEL-Passage-Encoder-Warmup \
  --save_steps 20000 \
  --data_dir beir \
  --train_path sent-query-t5-base-hard-negatives-iteration1.jsonl \
  --corpus_file tokenized_passage.jsonl \
  --query_file tokenized_query.jsonl \
  --fp16 \
  --per_device_train_batch_size 128 \
  --negatives_x_device \
  --multi_task \
  --noisy \
  --noise_type sequential \
  --noise_prob 0.1 \
  --shared_encoder False \
  --train_n_passages 2 \
  --max_query_length 128 \
  --max_passage_length 256 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --dataloader_num_workers 0
