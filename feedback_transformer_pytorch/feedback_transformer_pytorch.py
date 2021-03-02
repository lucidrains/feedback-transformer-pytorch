import math
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

# constants

Memory = namedtuple('Memory', ['keys', 'values'])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def safe_cat(arr, el, dim = 1):
    if not exists(arr):
        return el
    return torch.cat((arr, el), dim = dim)

# positional embedding

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        causal = False,
        num_buckets = 32,
        max_distance = 128,
        heads = 8
    ):
        super().__init__()
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return bias

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class SkipIf(nn.Module):
    def __init__(self, cond, fn):
        super().__init__()
        self.cond = cond
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        if self.cond(x, *args, **kwargs):
            return x
        return self.fn(x, *args, **kwargs)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, pos_emb = None):
        h, n, device = self.heads, x.shape[1], x.device

        self_attend = n > 1 # only self attend if going at greater than 1 token at a time

        q = self.to_q(x) * self.scale

        k, v = memory if exists(memory) else (None, None)

        if self_attend:
            self_k, self_v = self.to_kv(x).chunk(2, dim = -1)
            k = safe_cat(k, self_k, dim = 1)
            v = safe_cat(v, self_v, dim = 1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        i, j = sim.shape[-2:]

        if exists(pos_emb):
            sim = sim + pos_emb(sim)

        if self_attend:
            causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            causal_mask = rearrange(causal_mask, 'i j -> () () i j')
            mask_value = -torch.finfo(q.dtype).max
            sim.masked_fill_(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main class

class FeedbackTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        mem_len,
        seq_len = 2,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        keep_last_hidden = False
    ):
        super().__init__()
        self.seq_len = seq_len
        self.mem_len = mem_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = RelativePositionBias(causal = True, heads = heads)

        # main layers

        self.layers = nn.ModuleList([])
        shared_kv_proj = None

        for _ in range(depth):
            attn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            ff = FeedForward(dim = dim, dropout = ff_dropout)

            shared_kv_proj = default(shared_kv_proj, attn.to_kv)
            attn.to_kv = shared_kv_proj

            attn, ff = map(lambda fn: Residual(PreNorm(dim, fn)), (attn, ff))

            if seq_len == 1:
                memory_is_empty = lambda *args, **kwargs: not exists(kwargs['memory'])
                attn = SkipIf(memory_is_empty, attn)

            self.layers.append(nn.ModuleList([
                attn,
                ff
            ]))

        # memory parameters

        self.layer_weight = nn.Parameter(torch.ones(depth + 1))
        self.shared_kv_proj = shared_kv_proj
        self.keep_last_hidden = keep_last_hidden

        # final projection to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, memory = None, return_memory = False):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)

        memory_keys = None
        memory_values = None

        if exists(memory):
            memory_keys, memory_values = memory

        outputs = []

        # calculate weighting of layers for storing to memory

        layer_weight = self.layer_weight.softmax(dim = -1)
        layer_weight = rearrange(layer_weight, 'd -> d () () ()')

        for x in x.split(self.seq_len, dim = 1):
            hiddens = [x]

            # prepare memory for attention, if it exists

            memory = None
            if exists(memory_keys):
                memory = (memory_keys, memory_values)

            for attn, ff in self.layers:

                x = attn(x, memory = memory, pos_emb = self.pos_emb)
                x = ff(x)

                hiddens.append(x)

            outputs.append(x)

            # calculate new memory key / values and store to FIFO queue

            if self.keep_last_hidden: # secret option for only keeping last hidden layer, as in paper
                agg_hiddens = hiddens[-1]
            else:
                hiddens = torch.stack(hiddens)
                agg_hiddens = (hiddens * layer_weight).sum(dim = 0)

            # pre-calculate memory key / values and store to buffer

            mem_k, mem_v = self.shared_kv_proj(agg_hiddens).chunk(2, dim = -1)
            memory_keys = safe_cat(memory_keys, mem_k, dim = 1)
            memory_values = safe_cat(memory_values, mem_v, dim = 1)

            # enforce max length on memory buffer

            memory_keys = memory_keys[:, -self.mem_len:]
            memory_values = memory_values[:, -self.mem_len:]

        x = torch.cat((outputs), dim = 1)
        out = self.to_logits(x)

        if not return_memory:
            return out

        return out, Memory(memory_keys, memory_values)
