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

x = torch.randint(0, 256, (2, 512)).cuda()
model(x)  # (1, 512, 20000)
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
