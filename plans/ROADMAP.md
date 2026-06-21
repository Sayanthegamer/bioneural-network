# North Star Implementation Roadmap

## 1. Project Phasing and Milestones
[cite_start]This roadmap executes the Metaplastic Neuro-Channel (MNC) framework across four distinct milestones[cite: 417]. [cite_start]Each stage has a strict deliverable that must be verified before proceeding to the next [cite: 419-420].

| Phase | Strategic Scope | Verification Milestone |
| :--- | :--- | :--- |
| **Phase 1: Custom Kernels** | [cite_start]Develop the core multiplication-free operators (Smooth LSE, HTDR, Adder surrogate) as custom Autograd/Triton kernels[cite: 419]. | [cite_start]Verification of identical PyTorch outputs with zero floating-point multiplications[cite: 419]. |
| **Phase 2: Metaplastic Training** | [cite_start]Implement variational Gaussian parameter tracking ($\mu, \sigma$) and the local MESU/cascade update loops[cite: 419]. | [cite_start]Verification of uncertainty-scaled parameter updates and bidirectional cascade state coupling [cite: 419-420]. |
| **Phase 3: Feature Ingestion** | [cite_start]Build the sentence chunking and local frozen embedding (`all-MiniLM-L6-v2`) extraction pipeline[cite: 420]. | [cite_start]Demonstration of real-time text processing and sequential, unbatched vector ingestion on a laptop CPU[cite: 420]. |
| **Phase 4: Protocol Auditing** | [cite_start]Run the complete 10-day delayed recall validation tests and capture system telemetry[cite: 420]. | [cite_start]Delivery of the final validation report against the Transformer control, certifying a complete pass[cite: 420]. |

## 2. Recommended Tech Stack
[cite_start]To ensure lightweight, local execution on commodity laptop hardware without massive framework overhead, the build must utilize the following stack [cite: 440-441]:
* [cite_start]**Core Engine:** PyTorch (v2.1+) using the native CPU vectorization backend and MPS (Metal Performance Shaders) for Apple M-series GPUs[cite: 442].
* [cite_start]**Kernel Compilation:** Triton (for Linux/WSL) or `pybind11` coupled with a modern C++ compiler (supporting AVX2/AVX-512) to bypass the Python interpreter overhead for the custom forward/backward passes[cite: 443].
* [cite_start]**Feature Extraction:** `sentence-transformers` (configured strictly with `all-MiniLM-L6-v2`) executing solely on CPU to generate 384-dimensional vectors[cite: 444].
* [cite_start]**Telemetry:** `psutil` and native PyTorch profiling hooks to log VRAM footprint and execution latency[cite: 445].

## 3. Strategic Risk Register
These are the known architectural hazards. The codebase must explicitly implement their corresponding mitigations.

### Risk A: The Software Lottery Overhead (High Impact)
* [cite_start]**Hazard:** Custom clamping operations written in native Python will run slower than PyTorch's highly optimized `matmul` C++ backend, creating a false-negative on the efficiency test [cite: 423-424].
* [cite_start]**Mitigation:** The custom `torch.autograd.Function` classes must eventually be compiled directly into native vectorized CPU instructions using Triton or C++ extensions[cite: 425].

### Risk B: Gradient Saturation in Deep Layers (Moderate Impact)
* [cite_start]**Hazard:** Hard minimum operators ($min(|x|, |w|)$) create zero-gradients when the input is smaller than the channel width, freezing the network [cite: 426-428].
* [cite_start]**Mitigation:** The mathematical primitive must use the continuous Log-Sum-Exp (LSE) approximation, and the parallel Chemical Neurotransmitter Bypass ($B(x,n)$) must be maintained to guarantee a differentiable signal pathway [cite: 429-434].

### Risk C: Variational Posterior Saturation (High Impact)
* [cite_start]**Hazard:** Over long, continuous data streams, the parameter variances ($\sigma^2$) will collapse to zero, permanently freezing the network's ability to learn (catastrophic remembering) [cite: 435-438].
* [cite_start]**Mitigation:** The bounded-memory variational prior from the BiMU framework must be applied to continuously pull variances back toward a non-zero baseline ($\sigma_{\text{prior}}$)[cite: 439].

## 4. Scientific Validation & Protocol Auditing Focus (Phase 3 & 4)

All initial Phase 3/4 validation concerns and the **6-part Empirical Characterization & Stress-Testing Suite** have been completed and verified:
1.  **Representational Capacity & Interference Hardness:** **Resolved** (Study 2 & `parameter_overlap.py`), showing orthogonal gradients ($\cos(\theta) \approx -0.02$) and sparse bottleneck overlap.
2.  **Gradient Step Asymmetry & Capacity Walls:** **Resolved** (`capacity_wall.py`), revealing a power-law recall degradation ($1/N$ recency-dominated overwrite) when facts scale up to 800.
3.  **Decoder-Only Transformer Control:** **Resolved** (Study 5), confirming the MNC matches parameter-scaled Transformer baselines.
4.  **u2 Cascade Restorative Pull Trajectory:** **Resolved** (Study 4 & `drift_analysis.py`), proving that $u_2$ acts as a stabilizer that halves parameter drift ($D_{\text{drift}} = 0.156$ vs. $0.243$ when disabled).
5.  **Distance Scaling Autocalibration:** **Resolved** and analytically derived.
6.  **Variance Telemetry Sweep:** **Resolved** (`variance_telemetry.py`), demonstrating tunable non-zero equilibria based on the decay rate $\alpha$.
7.  **Replay Complementarity:** **Resolved** (`replay_comparison.py`), showing that combining MESU's metaplastic gating with experience replay reduces forgetting by **12.8x** (to 5.2%), proving they are complementary.

## 5. Future Work & Next Steps

Following the characterization of MESU v1, future research should focus on:
1.  **Dynamic Prior Relaxation (`adaptive_alpha.py`):** Moving beyond time-based relaxation to gate prior unfreezing dynamically based on input novelty or semantic conflict.
2.  **Bottleneck Representational Capacity:** Addressing the L1 coordinate bottleneck by scaling the channel dimensions or exploring sparse modular sub-networks to prevent recency-dominated overwrite.
3.  **Vectorized Kernel Optimization:** Compiling the custom addition-based distance calculations into native Triton or C++ operations to maximize systems-level efficiency at the edge.