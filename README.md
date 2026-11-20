# Julia Autodiff CNN

> Reverse-mode automatic differentiation and a small CNN (conv, max-pool, dropout, dense) built from scratch in Julia, trained on Fashion-MNIST.

## What is inside

| File | Content |
|---|---|
| `flex.jl` | Computational graph, reverse-mode autodiff, operators (Dense, Conv2D, MaxPool2D, Dropout, ReLU, Softmax, cross-entropy), SGD optimizer, `Chain` model API, `DataLoader` |
| `AWID-2026-CNN-flex.ipynb` | Training and evaluation on Fashion-MNIST using `flex.jl` |

Architecture: `Conv(3x3, 1=>6) -> MaxPool(2) -> Conv(3x3, 6=>16) -> MaxPool(2) -> Dense(784=>84, relu) -> Dropout(0.4) -> Dense(84=>10)`

## Results

| Epoch | Train acc | Test acc |
|---|---|---|
| 1 | 85.65% | 84.82% |
| 2 | 86.70% | 85.73% |
| 3 | 87.41% | 86.37% |

## Run

```julia
using Pkg; Pkg.add(["MLDatasets", "CairoMakie", "IJulia"])
```

Open `AWID-2026-CNN-flex.ipynb` in Jupyter with a Julia kernel (tested on Julia 1.12).

---

Built for the AWID course (Applied Computer Science, Warsaw University of Technology), 2026.
