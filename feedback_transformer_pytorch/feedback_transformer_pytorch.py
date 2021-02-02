import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# helpers

def exists(val):
    return val is not None

# classes

class FeedbackTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        mem_len,
        seq_len = 1,
        heads = 8,
        dim_head = 64,
    ):
        super().__init__()

    def forward(self, x):
        return x
