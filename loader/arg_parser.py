from typing import Optional, Dict, Any, cast
import sys
import os

from transformers.hf_argparser import DataClassType, DataClass

from train.common.my_argument_parser import MyArgumentParser


def parse_args(dataclass_types: DataClassType,
               args: Optional[Dict[str, Any]] = None,
               error_to_exception: Optional[bool] = False) -> DataClass:
    parser = MyArgumentParser(dataclass_types, error_to_exception)
    if args is not None:
        return parser.parse_dict(args)[0]

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))[0]

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))[0]

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    return parsed_args[0]
