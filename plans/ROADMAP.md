# North Star Implementation Roadmap

## 1. Project Phasing and Milestones
This roadmap executes the Metaplastic Neuro-Channel (MNC) framework across four distinct milestones. Each stage has a strict deliverable that must be verified before proceeding to the next.

| Phase | Strategic Scope | Verification Milestone |
| :--- | :--- | :--- |
| **Phase 1: Custom Kernels** | Develop the core multiplication-free operators (Smooth LSE, HTDR, Adder surrogate) as custom Autograd/Triton kernels. | Verification of identical PyTorch outputs with zero floating-point multiplications. |
| **Phase 2: Metaplastic Training** | Implement variational Gaussian parameter tracking ($\mu, \sigma$) and the local MESU/cascade update loops. | Verification of uncertainty-scaled parameter updates and bidirectional cascade state coupling. |
| **Phase 3: Feature Ingestion** | Build the sentence chunking and local frozen embedding (`all-MiniLM-L6-v2`) extraction pipeline. | Demonstration of real-time text processing and sequential, unbatched vector ingestion on a laptop CPU. |
| **Phase 4: Protocol Auditing** | Run the complete 10-day delayed recall validation tests and capture system telemetry. | Delivery of the final validation report against the Transformer control, certifying a complete pass. |

## 2. Recommended Tech Stack
To ensure lightweight, local execution on commodity laptop hardware without massive framework overhead, the build must utilize the following stack:
* **Core Engine:** PyTorch (v2.1+) using the native CPU vectorization backend and MPS (Metal Performance Shaders) for Apple M-series GPUs.
* **Kernel Compilation:** Triton (for Linux/WSL) or `pybind11` coupled with a modern C++ compiler (supporting AVX2/AVX-512) to bypass the Python interpreter overhead for the custom forward/backward passes.
* **Feature Extraction:** `sentence-transformers` (configured strictly with `all-MiniLM-L6-v2`) executing solely on CPU to generate 384-dimensional vectors.
* **Telemetry:** `psutil` and native PyTorch profiling hooks to log VRAM footprint and execution latency.

## 3. Strategic Risk Register
These are the known architectural hazards. The codebase must explicitly implement their corresponding mitigations.

### Risk A: The Software Lottery Overhead (High Impact)
* **Hazard:** Custom clamping operations written in native Python will run slower than PyTorch's highly optimized `matmul` C++ backend, creating a false-negative on the efficiency test.
* **Mitigation:** The custom `torch.autograd.Function` classes must eventually be compiled directly into native vectorized CPU instructions using Triton or C++ extensions.

### Risk B: Gradient Saturation in Deep Layers (Moderate Impact)
* **Hazard:** Hard minimum operators ($min(|x|, |w|)$) create zero-gradients when the input is smaller than the channel width, freezing the network.
* **Mitigation:** The mathematical primitive must use the continuous Log-Sum-Exp (LSE) approximation, and the parallel Chemical Neurotransmitter Bypass ($B(x,n)$) must be maintained to guarantee a differentiable signal pathway.

### Risk C: Variational Posterior Saturation (High Impact)
* **Hazard:** Over long, continuous data streams, the parameter variances ($\sigma^2$) will collapse to zero, permanently freezing the network's ability to learn (catastrophic remembering).
* **Mitigation:** The bounded-memory variational prior from the BiMU framework must be applied to continuously pull variances back toward a non-zero baseline ($\sigma_{\text{prior}}$).

## 4. Scientific Validation & Protocol Auditing Focus (Phase 3 & 4)

All initial Phase 3/4 validation concerns and the **8-part Empirical Characterization & Stress-Testing Suite** have been completed and verified:
1.  **Representational Capacity & Interference Hardness:** **Resolved** (Study 2 & `parameter_overlap.py`), showing orthogonal gradients ($\cos(\theta) \approx -0.02$) and sparse bottleneck overlap.
2.  **Gradient Step Asymmetry & Capacity Walls:** **Resolved** (`capacity_wall.py`), revealing a power-law recall degradation ($1/N$ recency-dominated overwrite) when facts scale up to 800.
3.  **Decoder-Only Transformer Control:** **Resolved** (Study 5), confirming the MNC matches parameter-scaled Transformer baselines.
4.  **u2 Cascade Restorative Pull Trajectory:** **Resolved** (Study 4 & `drift_analysis.py`), proving that $u_2$ acts as a stabilizer that reduces active drift by **35.7%** ($D_{\text{drift}} = 0.156$ vs. $0.243$ when disabled, or by **53%** from $W_0$: $0.120$ vs. $0.256$).
5.  **Distance Scaling Autocalibration:** **Resolved** and analytically derived.
6.  **Variance Telemetry Sweep:** **Resolved** (`variance_telemetry.py`), demonstrating tunable non-zero equilibria based on the decay rate $\alpha$.
7.  **Replay Complementarity:** **Resolved** (`replay_comparison.py`), showing that combining MESU's metaplastic gating with experience replay (size 10) reduces forgetting by **10.4x** (to 6.4%), and by **12.8x** (to 5.2%) with buffer size 50, proving they are complementary.
8.  **Parametric Isolation Sweep (`parametric_study.py`)**: **Resolved**, validating 100% linear and 99.34% MLP offline separability (confirming representational geometry), while isolating catastrophic online learning failures ($7.88\%$ online linear, $0.16\%$ online MLP).
9.  **Frozen Representation Test (Audit 3)**: **Resolved** (`check_frozen_rep.py`), demonstrating that output template interference is the primary driver of catastrophic forgetting.
10. **Replay Primacy Bias**: **Resolved**, diagnosing the logit domination of early classes in sequential experience replay.
11. **Bottleneck Width-Scaling Extrapolation (W=512, N=1600):** **Resolved** (`prototypical_width_scaling.py`), confirming that representational volume capacity scales sub-linearly with bottleneck width ($N_{50} = 17.10 \cdot W^{0.81}$) and validating the geometric density collapse principle ($\text{crowding} \propto N/W$) under a frozen prototypical readout network.
12. **Cross-Encoder Replication Study:** **Resolved** (`cross_encoder_study.py`), testing the universality of the $N/W$ density law across 5 semantic encoders (MiniLM, E5-small, MPNet, E5-base, BGE-small) and a Random Projection baseline. Confirmed **Outcome B** (manifold quality dependency): the density law shape is universal (all curves flatten at $W \ge 128$), but the plateau recall level depends on encoder-specific manifold quality. MiniLM (384d) dominates at ~78.5%, while 768d encoders (MPNet: ~43.8%, E5-base: ~38.7%) underperform due to sparser manifold structure under random projection. Decomposed the capacity law into $\text{Recall} = f(N/W) \cdot g(\mathcal{E})$.

## 5. Future Work & Next Steps

Following the characterization of MESU v1, future research should focus on:
1.  **Breaking the Prototype Assumption (Experiment 2):** Test whether the density wall is fundamental or a retrieval artifact by splitting clusters or using local exemplar neighborhoods instead of single global prototypes. If multi-prototype or local-exemplar methods raise the plateau recall for MiniLM above ~78.5%, the single-centroid assumption is the bottleneck, not geometry.
2.  **Noise/Robustness Testing (Experiment 3):** Evaluate recall under paraphrased, reordered, or synonym-substituted queries to test whether the current system is robust to realistic linguistic variation.
3.  **Out-of-Distribution/Compositional Retrieval (Experiment 4):** Test retrieval of compositionally novel facts that combine elements from multiple training statements.
4.  **True Continual Learning (Experiment 5):** Evaluate plasticity during representation updates — unfreezing the backbone and measuring how far prototypes drift.
5.  **Dynamic Prior Relaxation (`adaptive_alpha.py`):** Moving beyond time-based relaxation to gate prior unfreezing dynamically based on input novelty or semantic conflict.
6.  **Encoder-Aware Projection:** Replace random linear projection with a learned or SVD-based projection optimized for each encoder's manifold structure, targeting the $g(\mathcal{E})$ factor in the decomposed capacity law.
7.  **Vectorized Kernel Optimization:** Compiling the custom addition-based distance calculations into native Triton or C++ operations to maximize systems-level efficiency at the edge.