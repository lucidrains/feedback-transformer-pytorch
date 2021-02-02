## Feedback Transformer - Pytorch (wip)

Simple implementation of <a href="https://arxiv.org/abs/2002.09402">Feedback Transformer</a> in Pytorch. They improve on Transformer-XL by having each token have access to the representations of all previous layers through time. This is achieved by aggregating the outputs of all layers into a shared memory, which each token can attend to at each time step. The main drawback is longer training time, due to its non-parallel nature.

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
