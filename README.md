# Hands-on Transformer

An educational implementation of a small Transformer architecture using PyTorch.

## Project goal

Build a compact, CPU-friendly Transformer whose internal components are implemented explicitly and can be modified in experiments.

The first task is binary sentiment classification on movie reviews (IMDB). Later directions may include machine translation and biological sequence classification.

## How we work

The implementation is developed incrementally by the project author. The source file contains a small scaffold and TODO comments; components are added and tested one at a time. The numbered notebooks in the repository root are learning notes and may be edited independently.

## Repository structure

- `src/` — the main PyTorch implementation, currently a scaffold;
- `sandbox/` — the original JAX prototype supplied by the instructor. It is preserved unchanged;
- `01_*.ipynb`, `02_*.ipynb`, etc. — supporting learning notebooks.

## Planned first milestone

1. Implement token and positional embeddings.
2. Add the Transformer encoder block component by component.
3. Prepare the IMDB dataset and batching.
4. Train and evaluate a small sentiment classifier.
5. Compare model configurations and a TF-IDF + Logistic Regression baseline.

## Project dependencies

Install the packages from the repository root:

```bash
pip install -r requirements.txt
```

## References

- [nanoGPT — Andrej Karpathy](https://github.com/karpathy/nanoGPT)
- [PyTorch](https://pytorch.org/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
