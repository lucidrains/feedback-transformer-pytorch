## Feedback Transformer - Pytorch

Simple implementation of <a href="https://arxiv.org/abs/2002.09402">Feedback Transformer</a> in Pytorch. They improve on Transformer-XL by having each token have access to the representations of all previous layers through time. This is achieved by aggregating the outputs of all layers into a shared memory, which each token across layers can attend to at each time step.

The main drawback is longer training time, due to its non-parallel nature. But I thought I'd build it to further exploration and research into this line of work.

<a href="https://www.youtube.com/watch?v=zdb8MM94A5c">Yannic Kilcher video</a>

I also took the liberty to add some various enhancements, including pre-normalization, GLU gated feedforwards, as well as simplified T5 relative positional embeddings.

## Install

```bash
$ pip install feedback-transformer-pytorch
```

## Usage

```python
import torch
from feedback_transformer_pytorch import FeedbackTransformer

model = FeedbackTransformer(
    num_tokens = 20000,           # number of tokens
    dim = 512,                    # dimension
    depth = 6,                    # depth
    seq_len = 2,                  # the sequence length of each segment or window
    mem_len = 256,                # length of the memory buffer
    dim_head = 64,                # dimension of each head
    heads = 8,                    # number of heads
    attn_dropout = 0.1,           # attention dropout
    ff_dropout = 0.1              # feedforward dropout
).cuda()

x = torch.randint(0, 20000, (2, 64)).cuda()
model(x)  # (2, 64, 20000)
```

If you would like to have fine control over the memory (when to detach, etc), you can do it with some extra keyword arguments on `.forward`

```python
import torch
from feedback_transformer_pytorch import FeedbackTransformer

model = FeedbackTransformer(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    seq_len = 32,
    mem_len = 256
).cuda()

x1 = torch.randint(0, 20000, (2, 32)).cuda()
x2 = torch.randint(0, 20000, (2, 32)).cuda()
x3 = torch.randint(0, 20000, (2, 32)).cuda()

out1, mem1 = model(x1, return_memory = True)
out2, mem2 = model(x2, memory = mem1, return_memory = True)
out3, mem3 = model(x3, memory = mem2, return_memory = True)  # (2, 32, 20000)
```

## Cross attention

```python
import torch
from feedback_transformer_pytorch import FeedbackTransformer

model = FeedbackTransformer(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    seq_len = 1,
    mem_len = 256,
    cross_attend = True
).cuda()

x1 = torch.randint(0, 20000, (2, 32)).cuda()
x2 = torch.randint(0, 20000, (2, 32)).cuda()
x3 = torch.randint(0, 20000, (2, 32)).cuda()

encoded = torch.randn(2, 32, 512).cuda()

out1, mem1 = model(x1, context = encoded, return_memory = True)
out2, mem2 = model(x2, context = encoded, memory = mem1, return_memory = True)
out3, mem3 = model(x3, context = encoded, memory = mem2, return_memory = True)  # (2, 32, 20000)
```

## Citations

```bibtex
@misc{fan2021addressing,
    title   = {Addressing Some Limitations of Transformers with Feedback Memory}, 
    author  = {Angela Fan and Thibaut Lavril and Edouard Grave and Armand Joulin and Sainbayar Sukhbaatar},
    year    = {2021},
    eprint  = {2002.09402},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
