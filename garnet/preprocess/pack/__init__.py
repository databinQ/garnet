# coding: utf-8
# @Author: garnet
# @Time: 2019/12/4 23:17

import dill
import codecs
import typing
from tqdm import tqdm
from pathlib import Path


class DataPack(object):
    """
    Data structure, store data used by NLP tasks. In most cases, data will be stored in `pandas.DataFrame` structure.
    """
    DATA_FILENAME = "data.dill"

    @property
    def has_label(self) -> bool:
        raise NotImplemented

    def save(self, dir_path: typing.Union[str, Path]):
        dir_path = Path(dir_path)
        file_path = dir_path.joinpath(self.DATA_FILENAME)

        if file_path.exists():
            print("{} already exist, now covering old version data...")
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

        dill.dump(self, codecs.open(file_path, "wb"))

    @staticmethod
    def _apply_on_text(data, func: typing.Callable, text_col, name="applied_text", verbose=1):
        if verbose:
            tqdm.pandas(desc="Processing " + name + " with " + func.__name__)
            data[name] = data[text_col].progress_apply(func)
        else:
            data[name] = data[text_col].apply(func)
        return data

    def apply(self, func: typing.Callable, mode="normal", name="applied_text", verbose=1):
        """
        Apply function(s) to text columns, results are saved in `name` column
        """
        raise NotImplemented


class ModelDataPack(DataPack):
    """
    Data stored in this instance will be used for training models or testing. Thus, data is organized in proper format.
    """
    pass
