import functools
from typing import Mapping, Sequence, Optional, Dict, Any

from datasets import load_dataset

from model_wrapper.internlm2_tokenizer_wrapper_2 import InternLM2TokenizeWrapper_2
from train.loader.data.data_args import DataArgs


def process_batch(
        batch: Mapping[str, Sequence],
        max_length,
        tokenizer,
) -> dict[str, list]:
    ret = InternLM2TokenizeWrapper_2(tokenizer).batch_prepare_data_ft(batch, max_length, False)
    # ret["attention_mask"] = ret["input_ids"].ne(tokenizer.pad_token_id)
    return ret


def get_dataset2(tokenizer, args: Optional[Dict[str, Any]] = None):

    data_args = DataArgs.get(args)
    ds = load_dataset("json",
                      data_files=data_args.train_data_path,
                      split="train")

    map_func = functools.partial(
        process_batch,
        max_length=data_args.cutoff_len,
        tokenizer=tokenizer
    )
    return ds.map(map_func, num_proc=16, batched=True, remove_columns=["tools", "messages"])
