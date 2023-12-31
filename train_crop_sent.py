import logging
import os
import sys

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from arguments import ModelArguments, DataArguments, BiEncoderTrainingArguments
from dataloader import BiEncoderDataset, BiEncoderCollator, GenericDataLoader, NoisyBiEncoderDataset
from model import BiEncoder
from trainer import CropSentenceTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, BiEncoderTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: BiEncoderTrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    model = BiEncoder.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    train_datasets = []
    tasks = ['fiqa', 'scifact', 'arguana', 'climate-fever', 'dbpedia-entity', 'cqadupstack', 'quora', 'scidocs',
             'nfcorpus', 'signal1m', 'trec-covid', 'webis-touche2020', 'hotpotqa', 'nq', 'robust04', 'trec-news',
             'bioasq']
    data_dir = data_args.train_dir
    pretokenized = True
    use_mmap = True
    for task in tasks:
        data_dir = os.path.join(data_dir, task)

        corpus, queries, qrels = GenericDataLoader(data_dir, corpus_file=data_args.corpus_file,
                                                   query_file=data_args.query_file, qrel_file=data_args.train_path,
                                                   use_mmap=use_mmap).load(split="train")

        if training_args.noisy:
            train_dataset = NoisyBiEncoderDataset(corpus, queries, qrels, tokenizer, data_args,
                                                  pretokenized=pretokenized, noise_type=training_args.noise_type,
                                                  noise_prob=training_args.noise_prob)
        else:
            train_dataset = BiEncoderDataset(corpus, queries, qrels, tokenizer, data_args, pretokenized=pretokenized)
        train_datasets.append(train_dataset)

    data_collator = BiEncoderCollator(
        tokenizer,
        max_passage_length=data_args.max_passage_length,
        max_query_length=data_args.max_query_length
    )

    trainer = CropSentenceTrainer(
        model=model,
        train_dataset=train_datasets,
        data_collator=data_collator,
        training_args=training_args,
        data_args=data_args,
        tokenizer=tokenizer
    )

    for dataset in train_datasets:
        dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
