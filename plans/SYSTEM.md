# SYSTEM.md - Repository Rules and Context

## Project Definition
This repository contains the Metaplastic Neuro-Channel (MNC) Framework. It is a research initiative investigating bounded continual learning via metaplastic synaptic uncertainty.

## Current Project Status
Following comprehensive multi-seed validation, the core MESU optimizer and dual-timescale cascades have been verified as functional continual learning regularizers that outperform standard SGD on sequential tasks. However, scaling tests have identified major bottlenecks (e.g. embedding compute overhead, L1 template coordinate overlap, and memory overhead of the variance tracking states). The framework is currently maintained as a stable research checkpoint. 

## Immutable Coding Constraints (Read Carefully)
If you are contributing to or editing this codebase, you MUST adhere to the following rules:

1.  **NO MATRIX MULTIPLICATION IN THE CORE:** The forward pass of the core perceptron must never use `torch.matmul`, `F.linear`, or standard `nn.Linear`. Signals are routed exclusively via distance-based addition operators (production baseline uses the single-path `MNCAdderFunction` from `mnc/kernels.py`). The dual-path configuration is deferred for future research.
2.  **CUSTOM AUTOGRAD IS MANDATORY:** Do not attempt to rely on PyTorch's native automatic differentiation for the core perceptron distance operations. The equations require specialized mathematical approximations (such as the $L_2$ surrogate gradient on weights, and HTDR on inputs) to prevent dead gradients. These are housed in custom `torch.autograd.Function` classes (currently `MNCAdderFunction`). Any future dual-path or `SmoothMinLSE` implementation must also reside in a custom autograd class.
3.  **NO BATCHED TRAINING FOR EVALUATION:** The primary goal of this architecture is lifelong continual learning without a replay buffer. During the journal ingestion evaluation, the training loop must operate sequentially (`batch_size = 1`). Do not introduce standard `DataLoaders` that shuffle or repeat historical data.
4.  **HARDWARE TARGET:** This code is optimized for local execution on standard consumer CPUs/GPUs. Memory efficiency (avoiding intermediate tensor materialization) is prioritized.

## Glossary of Custom Mechanics
* **HTDR (Hard Threshold Derivative Replacement):** Used to clip gradients passed backward to input tensors to prevent chain-rule explosion.
* **LSE Smooth Minimum:** A differentiable approximation of `min(|x|, |w|)` to prevent zero-gradients.
* **MESU (Metaplasticity from Synaptic Uncertainty):** The Bayesian mechanism that locks down confident weights to prevent catastrophic forgetting.