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

All six initial Phase 3/4 validation concerns have been resolved and evaluated in [run_comprehensive_validation.py](file:///c:/Users/Anon/Downloads/Northstar/mnc_project/run_comprehensive_validation.py):
1. **Representational Capacity & Interference Hardness:** **Resolved** in Study 2 (shared output labels), demonstrating that MESU retains facts with only 0.50 average displaced facts compared to total SGD collapse.
2. **Gradient Step Asymmetry Control:** **Resolved** in Study 3 (symmetric budget sweeps), confirming that while asymmetry helps, MESU maintains recall advantages even under symmetric short budgets, though long symmetric interference degrades recall.
3. **Decoder-Only Transformer Control:** **Resolved** in Study 5, comparing against a tuned 5.2M parameter Transformer Decoder baseline.
4. **u2 Cascade Restorative Pull Trajectory:** **Resolved** in Study 4, proving that the $u_2$ restorative pull bounds cascade drift ($D(u_2, W_5)$) to 2.65 vs. 4.56 when disabled.
5. **Distance Scaling Autocalibration:** **Resolved** and integrated dynamically from random sphere distance distributions at initialization.
6. **Rigorous Multi-Seed Validation:** **Resolved** by reporting statistics (mean, std, range) across a 10-seed sweep for all studies.

### Current Open Focus Areas & Research Caveats:

1. **The Kernel Mismatch (Production Baseline vs. Deferred Dual-Path):**
   - **Status:** All validated numbers generated so far use the single-path `MNCAdderFunction` prototype. The dual-path Physical Channel ($C$) + Chemical Bypass ($N$) with LSE smoothing described in Section 2 is deferred as a future research milestone, as changing the gradient landscape could materially alter MESU consolidation dynamics.
2. **Single-Seed Transformer Tuning Instability:**
   - **Observation:** The automated hyperparameter tuning sweep for the Transformer Decoder control baseline can exhibit high variance or sensitivity when tuned on a single seed (Seed 0), sometimes failing to find optimal parameters that generalize stably across other seeds.
3. **Displacement-Metric Interpretation Caveat (Study 2):**
   - **Observation:** In Study 2 (interference in shared coordinates), a "displaced" fact does not necessarily mean complete forgetting, but rather a representation shift in the shared coordinate space that requires careful semantic classification boundaries.
4. **Pass/Fail Baseline Calculation:**
   - **Status:** The verification protocol requires measuring accuracy right after Day 5 training to establish the baseline denominator for the formal $\ge 85\%$ recall metric on Day 10. This is cheap and must be tracked.