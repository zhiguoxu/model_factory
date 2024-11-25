from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any
from train.common.logging import get_logger
from train.loader.arg_parser import parse_args

logger = get_logger(__name__)


@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: Optional[str] = field(
        default=None,
        metadata={
            "help": """Name(s) of target modules to apply LoRA. \
                    Use commas to separate multiple modules. \
                    Use "all" to specify all the available modules. \
                    LLaMA choices: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                    BLOOM & Falcon & ChatGLM choices: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], \
                    Baichuan choices: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                    Qwen choices: ["c_attn", "attn.c_proj", "w1", "w2", "mlp.c_proj"], \
                    InternLM2 choices: ["wqkv", "wo", "w1", "w2", "w3"], \
                    Others choices: the same as LLaMA."""
        },
    )
    lora_bf16_mode: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to train lora adapters in bf16 precision."},
    )
    use_rslora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."}
    )


@dataclass
class FinetuneArgs(LoraArguments):
    r"""
    Arguments for finetune
    """

    stage: Optional[Literal["sft"]] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )

    finetune_type: Optional[Literal["lora", "full"]] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )

    use_llama_pro: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )

    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "The number of trainable layers for partial-parameter (freeze) fine-tuning."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.lora_alpha = self.lora_alpha or self.lora_rank * 2
        self.lora_target = split_arg(self.lora_target)
        self.additional_target = split_arg(self.additional_target)

        assert self.finetune_type in ["lora", "freeze", "full"], "Invalid fine-tuning method."

        if self.use_llama_pro and self.finetune_type == "full":
            raise ValueError("`use_llama_pro` is only valid for the Freeze or LoRA method.")

    @staticmethod
    def get(args: Optional[Dict[str, Any]] = None) -> "FinetuneArgs":
        finetune_args = parse_args(FinetuneArgs, args)
        return finetune_args


if __name__ == "__main__":
    ...
