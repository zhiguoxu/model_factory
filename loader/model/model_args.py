from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Dict, Any, List

import torch
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers.integrations import is_deepspeed_zero3_enabled

from train.common.logging import get_logger
from train.common.misc import get_current_device
from train.common.packages import is_flash_attn2_available
from train.loader.arg_parser import parse_args
from train.loader.data.data_args import DataArgs

logger = get_logger(__name__)


@dataclass
class ModelArgs:
    r"""
    Arguments for model
    """

    pretrained_model_name_or_path: str = field(
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )

    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapter weight or identifier from huggingface.co/models."},
    )

    create_new_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )

    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."},
    )
    quantization_type: Optional[Literal["fp4", "nf4"]] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."},
    )
    double_quantization: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in int4 training."},
    )

    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )

    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."},
    )

    shift_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."},
    )

    # https://huggingface.co/blog/Andyrasika/finetune-unsloth-qlora
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )

    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )

    resize_vocab: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )

    disable_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )

    upcast_layernorm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    )

    upcast_lmhead_output: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
    )

    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )

    print_param_status: Optional[bool] = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters in the model."},
    )

    def __post_init__(self):
        self.do_train = None
        self.torch_dtype = None
        self.attn_implementation = self.init_attn_implementation()
        self.device_map = None  # zero3 不支持
        self.low_cpu_mem_usage = not is_deepspeed_zero3_enabled()  # zero3 不支持
        if self.adapter_name_or_path is not None:  # support merging multiple lora weights
            self.adapter_name_or_path: List[str] = [path.strip() for path in self.adapter_name_or_path.split(",")]
        self.model_max_length = None

    def to_load_args(self):
        args = asdict(self)
        args["torch_dtype"] = self.torch_dtype
        args["attn_implementation"] = self.attn_implementation
        args["device_map"] = self.device_map
        args["low_cpu_mem_usage"] = self.low_cpu_mem_usage

        args.pop("flash_attn", None)
        args.pop("shift_attn", None)
        args.pop("use_unsloth", None)
        args.pop("rope_scaling", None)
        args.pop("export_dir", None)
        args.pop("do_train", None)
        args.pop("resize_vocab", None)
        args.pop("disable_gradient_checkpointing", None)
        args.pop("adapter_name_or_path", None)
        args.pop("create_new_adapter", None)
        args.pop("quantization_bit", None)
        args.pop("quantization_type", None)
        args.pop("double_quantization", None)
        args.pop("upcast_layernorm", None)
        args.pop("upcast_lmhead_output", None)
        args.pop("hf_hub_token", None)
        args.pop("print_param_status", None)
        return args

    def init_attn_implementation(self) -> str | None:
        if self.flash_attn:
            if is_flash_attn2_available():
                logger.info("Using FlashAttention-2 for faster training and inference.")
                return "flash_attention_2"
            else:
                logger.warning("FlashAttention2 is not installed.")
                return None
        else:
            return "eager"

    def init_device_map(self, do_train):
        if is_deepspeed_zero3_enabled():
            return None
        if do_train:
            return {"": get_current_device()}
        elif self.export_dir is not None:
            return {"": "cpu"}
        else:
            return "auto"

    @staticmethod
    def get(args: Optional[Dict[str, Any]] = None, training_args: Seq2SeqTrainingArguments = None) -> "ModelArgs":
        model_args: ModelArgs = parse_args(ModelArgs, args)
        if training_args:
            model_args.do_train = training_args.do_train
            model_args.device_map = model_args.init_device_map(training_args.do_train)
            model_args.torch_dtype = (
                torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)
            )

        try:
            data_args: DataArgs = parse_args(DataArgs, args, error_to_exception=True)
            model_args.model_max_length = data_args.cutoff_len
        except Exception as e:
            ...

        return model_args


if __name__ == "__main__":
    print(ModelArgs.get().to_load_args())
