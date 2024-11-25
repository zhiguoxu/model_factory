import logging
from typing import Optional, List, Dict, Any

import transformers
from transformers import DataCollatorForSeq2Seq, TrainerCallback, Seq2SeqTrainingArguments

from train.common.callbacks import LogCallback
from train.common.logging import get_logger
from train.common.misc import get_logits_processor
from train.loader.data.data_args import DataArgs
from train.loader.data.load_data import get_dataset2
from train.loader.data.util import split_dataset
from train.loader.model.generating_args import GeneratingArguments
from train.loader.model.model_args import ModelArgs
from train.loader.model.model_loader import load_model
from train.loader.model.training_args import get_training_args
from train.loader.tokenizer.tokenizer_loader import load_tokenizer
from train.sft.metric import ComputeMetrics
from train.sft.trainer import CustomSeq2SeqTrainer

logger = get_logger(__name__)


def run_sft(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List[TrainerCallback]] = None):
    training_args = get_training_args(args)
    log_training_device(training_args)

    # 将 transformers 内部 logger level 设置成 INFO
    if training_args.should_log:
        _set_transformers_logging()

    callbacks = [LogCallback()] if callbacks is None else callbacks
    tokenizer = load_tokenizer(args)
    # dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")
    dataset = get_dataset2(tokenizer, args)

    model_args = ModelArgs.get(args, training_args)
    model = load_model(tokenizer, model_args, args)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_args = DataArgs.get(args)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args),
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Keyword arguments for `model.generate`
    generating_args = GeneratingArguments.get(args)
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)


def _set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def log_training_device(training_args: Seq2SeqTrainingArguments):
    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1)
        )
    )


if __name__ == "__main__":
    run_sft()
