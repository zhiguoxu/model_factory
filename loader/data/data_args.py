from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from train.loader.arg_parser import parse_args


@dataclass
class DataArgs:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """

    train_data_path: str = field(
        metadata={
            "help": "data file dir or path for training."
        },
    )

    val_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "data file dir or path for validation."
        },
    )

    val_size: Optional[float] = field(
        default=0,
        metadata={"help": "Size of the development set, should be an integer or a float in range `[0,1)`."},
    )

    cutoff_len: Optional[int] = field(
        default=1024,
        metadata={"help": "The cutoff length of the model inputs after tokenization."},
    )

    streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )

    buffer_size: Optional[int] = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )

    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"},
    )

    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation."
        },
    )

    @staticmethod
    def get(args: Optional[Dict[str, Any]] = None) -> "DataArgs":
        return parse_args(DataArgs, args)
