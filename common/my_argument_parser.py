from typing import Union, Iterable, Optional

from transformers import HfArgumentParser
from transformers.hf_argparser import DataClassType


class MyArgumentParser(HfArgumentParser):

    def __init__(self, dataclass_types: Union[DataClassType, Iterable[DataClassType]],
                 error_to_exception: Optional[bool] = False, **kwargs):
        super().__init__(dataclass_types, **kwargs)
        self.error_to_exception = error_to_exception

    def error(self, message):
        if self.error_to_exception:
            raise Exception(message)
        super(MyArgumentParser, self).error(message)
