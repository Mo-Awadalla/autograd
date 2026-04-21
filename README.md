# Autograd — A Micrograd Re-Implementation

A reverse-mode automatic differentiation engine implemented in Python,
re-implemented from Andrej Karpathy's "The spelled-out intro to neural
networks and backpropagation: building micrograd" tutorial.

I re-implemented this project from a working understanding of backpropagation
(not copy-paste) to solidify my knowledge of how autograd systems operate
under the hood.

## Contents

- `MicroGrad.ipynb` — The autograd engine and an example MLP trained on a
  toy dataset.
- `engine.py` — The `Value` autograd engine (scalar + tensor operations),
  numerical gradient checker, and PyTorch benchmark utilities.
- `test_engine.py` — Tests for the engine, gradient checker, and benchmark.

## What the engine supports

- A `Value` class with forward and backward passes
- Operations: `+`, `-`, `*`, `/`, `**`, `exp`, `tanh`
- Dynamic computation graph construction via parent tracking
- Topologically-ordered backward pass with gradient accumulation
- A small MLP implementation (Neuron → Layer → MLP) on top of the engine
- Tensor operations via `TensorValue`: matmul, broadcasting, reshape, sum
- Numerical gradient checker using finite differences
- Benchmarking vs. PyTorch autograd on identical inputs

## What I learned

- How reverse-mode autodiff works mechanically, including topological ordering
- Why gradients must accumulate (not overwrite) for shared parameters
- The chain rule applied to composed operations in code
- Why dynamic graphs are more flexible but harder to optimize than static graphs
- How tensor operations decompose into scalar autograd nodes
- How finite differences validate analytical gradients

## Next steps

- [x] Extend the engine to support tensor operations (matmul, broadcasting)
- [x] Add a numerical gradient checker using finite differences
- [x] Benchmark gradient computation vs. PyTorch autograd on identical inputs

## Credits

Based on [Andrej Karpathy's micrograd tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0)
and his [original implementation](https://github.com/karpathy/micrograd). All educational
credit to him.
