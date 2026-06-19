# MNC Staged Implementation Plan

## Stage 1: Custom Autograd Kernels (The Foundation)
**Objective:** Implement the mathematical primitives without triggering PyTorch's native graph memory blowouts or gradient failures.
* **Task 1.1:** Build and verify `MNCAdderFunction` as a custom `torch.autograd.Function`. Implement the HTDR clipping for inputs and the $L_2$ surrogate gradient for the weights. (Completed: This **single-path** prototype is the verified baseline used for all Study 1–5 evaluations).
* **Task 1.2 (Deferred / Ongoing):** Build `SmoothMinLSE` as a `torch.autograd.Function` and the parallel Chemical Bypass. This is deferred as a future research milestone since LSE gradient shapes could alter variance-gating dynamics.

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