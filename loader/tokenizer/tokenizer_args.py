from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from transformers import Seq2SeqTrainingArguments

from train.loader.arg_parser import parse_args


@dataclass
class TokenizerArgs:
    r"""
    Arguments for tokenizer
    """

    pretrained_model_name_or_path: str = field(
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )

    use_fast: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )

    padding_side: Optional[str] = field(
        default="left",
        metadata={"help": "tokenizer padding size."},
    )

    @staticmethod
    def get(args: Optional[Dict[str, Any]] = None) -> "TokenizerArgs":
        tokenizer_args = parse_args(TokenizerArgs, args)
        return tokenizer_args
