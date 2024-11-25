import os
from typing import Optional, Dict, Any

from transformers import Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from train.common.logging import get_logger
from train.loader.arg_parser import parse_args
from train.loader.model.fintune_args import FinetuneArgs

logger = get_logger(__name__)


def get_training_args(args: Optional[Dict[str, Any]] = None) -> Seq2SeqTrainingArguments:
    training_args: Seq2SeqTrainingArguments = parse_args(Seq2SeqTrainingArguments, args)
    finetune_args = FinetuneArgs.get(args)
    if (
            training_args.local_rank != -1
            and training_args.ddp_find_unused_parameters is None
            and finetune_args.finetune_type == "lora"
    ):
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args.ddp_find_unused_parameters = False

    if (
            training_args.resume_from_checkpoint is None
            and os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    return training_args
