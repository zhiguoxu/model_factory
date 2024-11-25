from dataclasses import asdict
from types import MethodType
from typing import Optional, Dict, Any

from transformers import PreTrainedTokenizer, AutoTokenizer, PreTrainedTokenizerBase
from train.loader.tokenizer.tokenizer_args import TokenizerArgs


def patch_tokenizer(tokenizer: PreTrainedTokenizer) -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)


def load_tokenizer(args: Optional[Dict[str, Any]] = None) -> PreTrainedTokenizer:
    args = TokenizerArgs.get(args)
    tokenizer = AutoTokenizer.from_pretrained(**asdict(args), trust_remote_code=True)
    patch_tokenizer(tokenizer)
    return tokenizer


if __name__ == "__main__":
    print(load_tokenizer())
