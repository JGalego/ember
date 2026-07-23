# References

This project is a research implementation inspired by publicly discussed
energy-based modeling (EBM) concepts and by public, non-technical
descriptions of Logical Intelligence's Kona-1. It is an original
implementation of ideas from the public EBM literature, not a reproduction
of, or reverse-engineering attempt against, any proprietary system, and it
makes no claim of architectural equivalence to Kona-1 or any other
proprietary model. See the "Public Descriptions of Kona-1" section below for
exactly which sources were treated as background context only.

## Energy-Based Models

- LeCun, Y., Chopra, S., Hadsell, R., Huang, F., & Ranzato, M. *A Tutorial on
  Energy-Based Learning*.
  http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf
  — source for the square-square contrastive loss in `models/losses.py` and
  the general framing of inference as energy minimization over a candidate
  output (`models/optimizer.py`, `inference/solve.py`).
- LeCun, Y. *A Path Towards Autonomous Machine Intelligence*.
  https://openreview.net/forum?id=BZ5a1r-kVsf
- LeCun Lab publications: https://atcold.github.io/

## Contrastive Learning

- SimCLR. https://arxiv.org/abs/2002.05709
- MoCo. https://arxiv.org/abs/1911.05722
- InfoNCE (CPC). https://arxiv.org/abs/1807.03748
  — source for `info_nce_loss` in `models/losses.py`.

## Optimization

- PyTorch LBFGS documentation.
  https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
  — the `lbfgs` method in `models/optimizer.py`.
- Torch Compile. https://pytorch.org/get-started/pytorch-2.0/
  — the optional `training.compile` flag in `configs/training/default.yaml`.

## Graph Neural Networks

- PyTorch Geometric. https://pytorch-geometric.readthedocs.io/
  — `models/encoder.py`'s `GNNEncoder` is a small hand-rolled message-passing
  layer to avoid a heavy extra dependency; PyG is the natural upgrade path
  for production-grade GNN encoders.

## Deep Sets

- Zaheer, M. et al. *Deep Sets*. https://arxiv.org/abs/1703.06114
  — `models/energy_model.py`'s `DeepSetsEnergy`.

## Transformers

- Vaswani, A. et al. *Attention Is All You Need*.
  https://arxiv.org/abs/1706.03762
  — `models/encoder.py`'s `TransformerEncoder`, `models/decoder.py`'s
  `TransformerDecoder`, `models/energy_model.py`'s `TransformerEnergy`.

## Constraint Satisfaction

- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach*.
  — general framing for the CSP domains in `datasets/` (Sudoku, SAT, graph
  coloring, mazes as shortest-path search) and the baseline solvers in
  `benchmarks/`.

## Public Descriptions of Kona-1

These are background context only and were **not** used as implementation
specifications -- no proprietary code, weights, architecture details, or
internal documentation was consulted or reproduced. Everything implemented
here comes from the publicly documented EBM literature above.

- Logical Intelligence homepage: https://logicalintelligence.com/
- Business Wire announcement:
  https://www.businesswire.com/news/home/20260120751310/en/Logical-Intelligence-Introduces-First-Energy-Based-Reasoning-AI-Model-Signals-Early-Steps-Toward-AGI-Adds-Yann-LeCun-and-Patrick-Hillmann-to-Leadership
- Reddit discussion (community interpretation, not authoritative):
  https://www.reddit.com/r/singularity/comments/1qk8trt/what_lecuns_energybased_models_actually_are/
