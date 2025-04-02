from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    scan_mode: str = 'cumsum'
    layer_type: str = 'mamba'

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        self.dt_rank = math.ceil(self.d_model / 16) if self.dt_rank == 'auto' else self.dt_rank

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)
        assert self.layer_type in ["mamba","cross_mamba"], f"Received invalid layer_type {self.layer_type}"

