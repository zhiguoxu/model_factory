import math

import torch
from transformers import PretrainedConfig, AutoConfig

from train.common.logging import get_logger
from train.common.misc import infer_optim_dtype
from train.loader.model.model_args import ModelArgs
from train.loader.patches.llama_patch import apply_llama_patch

logger = get_logger(__name__)
SUPPORTED_CLASS_FOR_S2ATTN = ["llama"]


def load_config(args: ModelArgs):
    config = AutoConfig.from_pretrained(args.pretrained_model_name_or_path, trust_remote_code=True)
    patch_config(config, args)
    return config


def patch_config(config: PretrainedConfig, model_args: ModelArgs) -> None:
    if model_args.torch_dtype is None:  # priority: bf16 > fp16 > fp32
        model_args.torch_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    if getattr(config, "model_type", None) == "qwen":
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.torch_dtype == dtype)

    if model_args.rope_scaling is not None:
        _configure_rope(config, model_args)

    if model_args.do_train and model_args.shift_attn:
        _configure_longlora(config)


def _configure_rope(config: PretrainedConfig, model_args: ModelArgs) -> None:
    if not hasattr(config, "rope_scaling"):
        logger.warning("Current model does not support RoPE scaling.")
        return

    if model_args.do_train:
        if model_args.rope_scaling == "dynamic":
            logger.warning(
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )

        current_max_length = getattr(config, "max_position_embeddings", None)
        if current_max_length and model_args.model_max_length > current_max_length:
            scaling_factor = float(math.ceil(model_args.model_max_length / current_max_length))
        else:
            logger.warning("Input length is smaller than max length. Consider increase input length.")
            scaling_factor = 1.0
    else:
        scaling_factor = 2.0

    setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
    logger.info(
        "Using {} scaling strategy and setting scaling factor to {}".format(model_args.rope_scaling, scaling_factor)
    )


def _configure_longlora(config: "PretrainedConfig") -> None:
    if getattr(config, "model_type", None) in SUPPORTED_CLASS_FOR_S2ATTN:
        setattr(config, "group_size_ratio", 0.25)
        apply_llama_patch()
        logger.info("Using shift short attention with group_size_ratio=1/4.")
    else:
        logger.warning("Current model does not support shift short attention.")
