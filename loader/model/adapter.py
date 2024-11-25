from typing import List, Optional, Dict, Any, cast

import torch
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
from transformers import PreTrainedModel
from transformers.integrations import is_deepspeed_zero3_enabled

from train.common.logging import get_logger
from train.loader.model.fintune_args import FinetuneArgs
from train.loader.model.model_args import ModelArgs

logger = get_logger(__name__)


def init_adapter(model: PreTrainedModel,
                 model_args: ModelArgs,
                 args: Optional[Dict[str, Any]] = None
                 ) -> PreTrainedModel:
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if (not model_args.do_train) and model_args.pretrained_model_name_or_path is None:
        logger.info("Adapter is not found at evaluation, load the base model.")
        return model

    finetune_args = FinetuneArgs.get(args)
    if finetune_args.finetune_type == "full" and model_args.do_train:
        logger.info("Fine-tuning method: Full")
        return model.float()

    logger.info("Fine-tuning method: {}".format("DoRA" if finetune_args.use_dora else "LoRA"))

    if model_args.adapter_name_or_path is None:
        model_args.create_new_adapter = model_args.do_train
    else:
        if not model_args.do_train:
            model_args.create_new_adapter = False   # 如果不是训练不需要 new_adapter

        if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            assert not model_args.create_new_adapter

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            assert not model_args.create_new_adapter

        if model_args.create_new_adapter:
            adapter_to_merge = model_args.adapter_name_or_path
            adapter_to_resume = None
        else:
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]

        for adapter in adapter_to_merge:
            model = cast(LoraModel, PeftModel.from_pretrained(model, adapter))
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info("Merged {} adapter(s).".format(len(adapter_to_merge)))

        if adapter_to_resume is not None:  # resume lora training
            model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=model_args.do_train)

    if model_args.create_new_adapter:  # create new lora weights while training
        if len(finetune_args.lora_target) == 1 and finetune_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model)
        else:
            target_modules = finetune_args.lora_target

        if finetune_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetune_args.num_layer_trainable)

        if finetune_args.use_dora:
            if getattr(model, "quantization_method", None):
                raise ValueError("DoRA is currently not compatible with quantized models.")

        peft_kwargs = {
            "r": finetune_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": finetune_args.lora_alpha,
            "lora_dropout": finetune_args.lora_dropout,
            "use_rslora": finetune_args.use_rslora,
        }

        if model_args.use_unsloth:
            from unsloth import FastLanguageModel  # type: ignore

            unsloth_peft_kwargs = {"model": model, "max_seq_length": model_args.model_max_length}
            model = FastLanguageModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                modules_to_save=finetune_args.additional_target,
                use_dora=finetune_args.use_dora,
                **peft_kwargs,
            )
            model = get_peft_model(model, lora_config)

    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.bfloat16 if finetune_args.lora_bf16_mode else torch.float32)

    if model_args.adapter_name_or_path is not None:
        logger.info("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    return model


def find_all_linear_modules(model: PreTrainedModel) -> List[str]:
    r"""
    Finds all available modules to apply lora.
    """
    quantization_method = getattr(model, "quantization_method", None)
    if quantization_method is None:
        linear_cls = torch.nn.Linear
    elif quantization_method == "bitsandbytes":
        import bitsandbytes as bnb

        linear_cls = bnb.nn.Linear4bit if getattr(model, "is_loaded_in_4bit", False) else bnb.nn.Linear8bitLt
    else:
        raise ValueError("Finding linear modules for {} models is not supported.".format(quantization_method))

    output_layer_names = ["lm_head"]
    if model.config.model_type == "chatglm":
        output_layer_names.append("output_layer")

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(output_layer in name for output_layer in output_layer_names):
            module_names.add(name.split(".")[-1])

    logger.info("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)


def find_expanded_modules(model: PreTrainedModel, target_modules: List[str], num_layer_trainable: int) -> List[str]:
    r"""
    Finds the modules in the expanded blocks to apply lora.
    """
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if not num_layers:
        raise ValueError("Model was not supported.")

    if num_layers % num_layer_trainable != 0:
        raise ValueError(
            "`num_layers` {} should be divisible by `num_layer_trainable` {}.".format(num_layers, num_layer_trainable)
        )

    stride = num_layers // num_layer_trainable
    trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
    trainable_layers = [".{:d}.".format(idx) for idx in trainable_layer_ids]
    module_names = []
    for name, _ in model.named_modules():
        if any(target_module in name for target_module in target_modules) and any(
            trainable_layer in name for trainable_layer in trainable_layers
        ):
            module_names.append(name)

    logger.info("Apply lora to layers: {}".format(",".join(map(str, trainable_layer_ids))))
    return module_names
