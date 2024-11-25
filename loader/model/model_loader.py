import math
from contextlib import nullcontext
from types import MethodType
from typing import Optional, Dict, Any, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, PretrainedConfig, AutoConfig
from transformers.integrations import is_deepspeed_zero3_enabled

from train.common.logging import get_logger
from train.common.misc import get_current_device, count_parameters
from train.loader.model.adapter import init_adapter
from train.loader.model.config_loader import load_config
from train.loader.model.model_args import ModelArgs

logger = get_logger(__name__)
LAYERNORM_NAMES = {"norm", "ln"}


def load_model(tokenizer: PreTrainedTokenizer,
               model_args: ModelArgs,
               args: Optional[Dict[str, Any]] = None,
               ) -> PreTrainedModel:
    config = load_config(model_args)
    model = load_by_unsloth(model_args, config)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(**model_args.to_load_args(), trust_remote_code=True, config=config)
    patch_model(model, tokenizer, model_args)
    register_autoclass(config, model, tokenizer)
    model = init_adapter(model, model_args, args)

    if model_args.do_train:
        model.train()
    else:
        model.requires_grad_(False)
        model = model.to(model_args.torch_dtype) if not getattr(model, "quantization_method", None) else model
        model.eval()

    logger.info(f"model torch_dtype: {model_args.torch_dtype}")
    log_model_param(model, model_args)

    return model


def load_by_unsloth(model_args: ModelArgs, config: AutoConfig):
    if not model_args.do_train or not model_args.use_unsloth:
        return None
    from unsloth import FastLanguageModel  # type: ignore

    unsloth_kwargs = {
        "model_name": model_args.pretrained_model_name_or_path,
        "max_seq_length": model_args.model_max_length,
        "dtype": model_args.torch_dtype,
        "load_in_4bit": model_args.quantization_bit == 4,
        "token": model_args.hf_hub_token,
        "device_map": {"": get_current_device()},
        "rope_scaling": getattr(config, "rope_scaling", None),
    }
    try:
        model, _ = FastLanguageModel.from_pretrained(**unsloth_kwargs)
    except NotImplementedError:
        logger.warning("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
        model_args.use_unsloth = False
        model = None

    if model_args.adapter_name_or_path:
        assert model_args.adapter_name_or_path is None, "Unsloth does not support loading adapters."

    return model


def _noisy_mean_initialization(embed_weight: torch.Tensor, num_new_tokens: int):
    embedding_dim = embed_weight.size(1)
    avg_weight = embed_weight[:-num_new_tokens].mean(dim=0, keepdim=True)
    noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
    noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
    embed_weight[-num_new_tokens:] = avg_weight + noise_weight


def _resize_embedding_layer(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer") -> None:
    r"""
    Resize token embeddings.
    """
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        params = [model.get_input_embeddings().weight]
        if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
            params.append(model.get_output_embeddings().weight)

        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    with context_maybe_zero3:
        current_embedding_size = model.get_input_embeddings().weight.size(0)

    if len(tokenizer) > current_embedding_size:
        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
            logger.warning("Current model does not support resizing token embeddings.")
            return

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        with context_maybe_zero3:
            new_embedding_size = model.get_input_embeddings().weight.size(0)
            num_new_tokens = new_embedding_size - current_embedding_size
            _noisy_mean_initialization(model.get_input_embeddings().weight.data, num_new_tokens)
            _noisy_mean_initialization(model.get_output_embeddings().weight.data, num_new_tokens)

        logger.info("Resized token embeddings from {} to {}.".format(current_embedding_size, new_embedding_size))


def _prepare_model_for_training(
        model: PreTrainedModel, model_args: ModelArgs, output_layer_name: Optional[str] = "lm_head"
) -> None:
    r"""
    Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads
        (3) add the upcasting of the lm_head in fp32
    Inspired by: https://github.com/huggingface/peft/blob/v0.7.1/src/peft/utils/other.py#L72
    """
    if model_args.upcast_layernorm:
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in LAYERNORM_NAMES):
                param.data = param.data.to(torch.float32)
        logger.info("Upcasting layernorm weights in float32.")

    if not model_args.disable_gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning("Current model does not support gradient checkpointing.")
        else:
            # use_reentrant=False might increase VRAM usage (have not been empirically verified yet)
            # According to: https://github.com/huggingface/transformers/issues/28339
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            model.enable_input_require_grads()
            model.config.use_cache = False  # turn off when gradient checkpointing is enabled
            logger.info("Gradient checkpointing enabled.")

    if hasattr(model, output_layer_name) and model_args.upcast_lmhead_output:

        def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
            return output.to(torch.float32)

        output_layer = getattr(model, output_layer_name)
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            output_layer.register_forward_hook(fp32_forward_post_hook)


def register_autoclass(config: PretrainedConfig, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    if "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()


def patch_model(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, args: ModelArgs
) -> None:
    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    if getattr(model.config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    if args.resize_vocab:
        _resize_embedding_layer(model, tokenizer)

    if args.do_train:
        _prepare_model_for_training(model, args)


def log_model_param(model: PreTrainedModel, model_args: ModelArgs):
    trainable_params, all_param = count_parameters(model)
    if model_args.do_train:
        param_stats = "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:d}".format(all_param)
    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )
