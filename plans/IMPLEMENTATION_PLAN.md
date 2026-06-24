# MNC Staged Implementation Plan

## Stage 1: Custom Autograd Kernels (The Foundation)
**Objective:** Implement the mathematical primitives without triggering PyTorch's native graph memory blowouts or gradient failures.
* **Task 1.1:** Build and verify `MNCAdderFunction` as a custom `torch.autograd.Function`. Implement the HTDR clipping for inputs and the $L_2$ surrogate gradient for the weights. (Completed: This **single-path** prototype is the verified baseline used for all Study 1–5 evaluations).
* **Task 1.2 (Deferred):** Build `SmoothMinLSE` as a `torch.autograd.Function` and the parallel Chemical Bypass. This is deferred as a future research milestone since LSE gradient shapes could alter variance-gating dynamics.

## Stage 2: The Core Network Module (Production Validation)
**Objective:** Assemble the single-path primitive into a trainable layer and run comprehensive multi-seed validation.
* **Task 2.1:** Integrate `MNCAdderFunction` and distance autocalibration into `MNCLinear` (Completed).
* **Task 2.2:** Run the comprehensive 10-seed validation suite (Studies 1–5) to verify the MESU vs SGD gap, step asymmetry, and $u_2$ consolidation pull under symmetric/asymmetric interference (Completed).

## Stage 3: The Data Ingestion Pipeline
**Objective:** Build the local text-processing pipeline for the delayed-recall test.
* **Task 3.1:** Integrate `sentence-transformers` (`all-MiniLM-L6-v2`) to execute strictly on local CPU (Completed).
* **Task 3.2:** Write a deterministic sentence chunker for the journal strings (Completed).
* **Task 3.3:** Build the sequential training loop. **Strict Constraint:** Batch size is 1. No historical storage arrays. Text is discarded immediately after the backward pass (Completed).

## Stage 4: Evaluation and Falsification
**Objective:** Run the 10-Day Delayed Recall Protocol.
* **Task 4.1:** Train the MNC sequentially on Days 1–9. Query on Day 10 (Completed).
* **Task 4.2:** Train a baseline Transformer ($\approx 5.2M$ params) on the exact same unbatched stream (Completed).
* **Falsification Check:**
  - MNC training latency is $<1.20\text{x}$ the baseline. (Passed).
  - The Transformer control baseline is tuned on its own terms (via LR and step sweeps). (Passed).
  - Report recall under both symmetric and asymmetric budgets, establishing immediate post-training Day 5 baselines to compute formal pass/fail ratios. (Completed).

## Stage 5 (Future Research): Dual-Path Channel + Bypass
**Objective:** Implement the dual-path LSE-smoothed model as a separate development branch.
* **Task 5.1:** Implement `SmoothMinLSE` and a somatic integration path summing both the physical channel and chemical bypass.
* **Task 5.2:** Re-run the comprehensive 10-seed validation sweep to check if the MESU vs SGD gap survives the smoothed gradient shape changes.

## Stage 6: Empirical Characterization & Stress-Testing Suite
**Objective:** Map the physical and optimization limits of the MESU v1 optimizer.
* **Task 6.1:** Implement u2 Cascade Ablation, Gradient Overlap, Multi-Decay Variance Telemetry, Drift Correlation, Capacity Wall Scaling, and Replay Comparison sweeps. (Completed).
* **Task 6.2:** Sweep representational capacity (model width) to construct log-log scaling fits under $\text{Recall}(N) = A/N^\alpha$. (Completed: Confirmed that forgetting is interference-dominated since the decay exponent remains constant at $\alpha \approx 1.20$ regardless of network width scaling).
* **Task 6.3:** Overhaul the physics identity-retrieval lab (`laboratory.py`) to correct Nearest Neighbor lookup logic and separate Geometry vs Success margins. (Completed).

## Stage 7: Parametric Isolation Study
**Objective:** Build a rigorous causal ladder separating representation geometry from continual learning optimization.
* **Task 7.1:** Implement the `parametric_study.py` sweep tracking offline controls (`OfflineLinear` and `OfflineMLP`) alongside online baseline models. (Completed).
* **Task 7.2:** Track and graph retention curves (Current vs. Probe recall) and generate concept density heatmaps. (Completed).

## Stage 8: Post-Run Audits & Resolutions
**Objective:** Investigate and resolve anomalies observed during parametric isolation.
* **Task 8.1:** Diagnose and resolve the `OfflineMLP` generalization gap by regularizing outputs and optimizing training convergence (removing early stopping limits). (Completed: MLP recall resolved to $99.34\% \pm 0.12\%$).
* **Task 8.2:** Audit `ReplayLinear` primacy bias (logit domination from early class updates). (Completed).
* **Task 8.3:** Implement the Frozen Representation Test to isolate hidden representation drift from output template interference. (Completed: Proved that output template interference is the primary driver of catastrophic forgetting).

## Stage 9: Relational Manifold Audit & Oracles (Experiment 3)
**Objective:** Run a rigorous diagnostic audit to trace semantic representational capacity and investigate metric crowding/information loss.
* **Task 9.1:** Build [relational_manifold_audit.py](file:///c:/Users/Anon/Downloads/Northstar/experiments/relational_manifold_audit.py) to evaluate 5 semantic encoders up to $N=800$ (and $N=1600$ for baseline MiniLM and best alternative). Track prototype advantage, Margins CDF, oracle centroid vs. exemplar Top-K recall, distance metrics (L1, L2, Cosine) on raw/L2-normalized embeddings, and projection distortion (Random vs SVD projection). (Script completed and running in background).
* **Task 9.2 (In Progress):** Speed up LinearSVC classification probe fits for large N (800 and 1600 facts) on normalized data. Current progress indicates that the primal solver (`dual=False`) takes $\approx 150$ seconds per configuration fit due to hypersphere packing density. Running benchmark [benchmark_svc.py](file:///C:/Users/Anon/.gemini/antigravity-ide/brain/a32e5034-4795-4ac6-abd7-7b8295d50bbb/scratch/benchmark_svc.py) to test dual solver (`dual=True` or `tol=1e-3`) for 10x–100x speedup.