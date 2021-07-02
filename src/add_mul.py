from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.data

from utils import split_digits

# This is taken wholesale from Robert Csordas' code investigating modularity in
# a different way: https://github.com/RobertCsordas/modules/blob/
# 3c9422bfd841a1a6aa3dd5e538f5a966b820df4f/dataset/double_op.py


class DoubleOpDataset(torch.utils.data.Dataset):
    DATA = {}
    OP_IDS: List[str]
    SETS = ["train", "test", "valid"]

    def __init__(self,
                 set: str,
                 n_samples: int,
                 n_digits: int,
                 restrict: Optional[List[str]] = None):
        super().__init__()
        self.n_digits = n_digits
        self.set = set
        self.full_name = (f"{self.__class__.__name__}_{set}" +
                          f"_{n_digits}_{n_samples}")
        self.classes = self.OP_IDS

        if self.full_name not in self.DATA:
            seed = np.random.RandomState(0x12345678 + self.SETS.index(set))
            self.DATA[self.full_name] = {
                "args": seed.randint(0, 10**n_digits, (n_samples, 2)),
                "op": seed.randint(0, 2, (n_samples, ))
            }

        self.data = self.DATA[self.full_name]
        if restrict:
            mask = False
            for r in restrict:
                mask = (self.data["op"] == self.OP_IDS.index(r)) | mask

            self.data = {
                "args": self.data["args"][mask],
                "op": self.data["op"][mask]
            }

    def out_channels(self) -> int:
        return self.n_digits * 10

    def in_channels(self) -> int:
        return self.n_digits * 2 * 10 + 2

    def __len__(self) -> int:
        return self.data["op"].shape[0]

    def __getitem__(self, item: int) -> Dict[str, np.ndarray]:
        args = self.data["args"][item]
        op = self.data["op"][item]
        res = self.get_res(item)

        return {
            "input": split_digits(self.n_digits, args),
            "output": split_digits(self.n_digits, res[op]),
            "op": np.array(op, dtype=np.uint8),
            "all_res": split_digits(self.n_digits, np.stack(res, 0))
        }


class AddMul(DoubleOpDataset):
    OP_IDS = ["add", "mul"]

    def get_res(self, item: int):
        args = self.data["args"][item]
        res = [args[0] + args[1], args[0] * args[1]]
        max = 10**self.n_digits
        return [r % max for r in res]
