# Metaplastic Neuro-Channel (MNC) Framework v1.0
## Architectural Blueprint

The MNC is a research-grade neural architecture designed to investigate bounded continual learning on local hardware. It replaces standard dot-product cross-correlations with L1 distance metrics and utilizes synaptic uncertainty tracking (MESU) alongside dual-timescale cascades to prevent parameter drift.

### 1. The Core Philosophy
The MNC is a hardware-native, multiplication-free neural architecture designed for lifelong, sequential learning on local consumer hardware. It discards standard cross-correlation (matrix multiplication) in favor of ALU-friendly flow-control operations, bypassing the computational bottlenecks of standard Transformers.

### 2. The Forward Primitive: Dual-Path Flow Control
The standard neuron dot-product is replaced by two parallel signal paths. [cite_start]To prevent the dead gradients associated with hard clamping operations, the physical channel is smoothed using a Stable Log-Sum-Exp (LSE) approximation[cite: 526, 572, 825].

* [cite_start]**Smoothed Absolute Values:** Let $A = \sqrt{x^2 + \epsilon}$ and $W = \sqrt{w^2 + \epsilon}$ (where $\epsilon = 10^{-8}$)[cite: 556, 558].
* [cite_start]**The Physical Channel ($C$):** $C(x, W) = \text{sgn}(x) \cdot \left[ \min(A,W) - \frac{1}{\alpha}\log(1 + \exp(-\alpha|A - W|)) \right]$ [cite: 572, 636]
* **The Chemical Bypass ($B$):** $B(x, N) = \text{sgn}(x) \cdot \left[ \min(A,N) - \frac{1}{\alpha}\log(1 + \exp(-\alpha|A - N|)) \right]$
* **Somatic Integration:** $y_j = \frac{1}{\sqrt{d_{in}}} \sum_{i=1}^{d_{in}} (C_i + B_i) + b_j$

### 3. The Backward Primitive: Decoupled Virtual Routing
[cite_start]Because absolute differences yield discrete, mathematically useless derivatives (the sign function), the backward pass relies on decoupled virtual gradient routing[cite: 907, 939, 953].

* [cite_start]**Weight Parameter Gradients ($L_2$ Surrogate):** Updates to the channel widths ignore the true derivative and route the full-precision difference $(X - W)$ backward to prevent oscillatory failure[cite: 955, 960].
* **Input Gradients (HTDR):** Gradients passed down the network chain are strictly clamped using HardTanh: $\text{HT}(W - X)$ bounded between $[-1, 1]$ to prevent chain-rule explosion[cite: 978, 983].
* [cite_start]**Adaptive Normalization:** Because addition yields smaller variances, layer weight updates must be dynamically scaled by $\eta \frac{\sqrt{k}}{||\Delta L(F_l)||_2}$[cite: 1009, 1026].

### 4. The Continual Memory Engine
The network learns sequentially without a replay buffer using Metaplasticity from Synaptic Uncertainty (MESU).

* **Uncertainty-Gated Plasticity:** Every parameter tracks its mean ($\mu$) and variance ($\sigma^2$). Update magnitudes are proportional to variance: $\Delta\mu \propto \sigma^2 \cdot \nabla_{\mu} \mathcal{L}$.
* **Prior Relaxation:** Variances are continuously pulled toward a baseline $\sigma_{\text{prior}}$ to prevent catastrophic remembering (permanent parameter freezing).
* **Multi-Timescale Cascades & Stabilization:** Fast execution variables ($u_1$) are bidirectionally coupled to slow consolidation reservoirs ($u_2$). Coupling conductance ($g$) is gated by the network's global predictive loss. The restorative pull anchors the slow cascade ($u_2$) near consolidated coordinates to bound long-term parameter drift into noise, though its short-term recall impact is modest.
* **Step Budget Asymmetry:** Asymmetric step budgeting (giving target facts a higher optimization budget than interference) acts as a necessary structural safeguard. The MESU uncertainty governor is necessary but not sufficient on its own to withstand high-energy, symmetric (equal-budget) interference.