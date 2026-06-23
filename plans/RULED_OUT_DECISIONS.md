# Ruled Out Decisions & Failed Experiments Log

This document tracks every design decision, loss function, optimizer configuration, and bug pattern that has been empirically tested and proven to be ineffective, sub-optimal, or discarded in the BioNeural Network (MNC) project. 

---

## ❌ 1. Gradient Rescaling / Width-Invariant Locking
*   **The Idea:** To prevent wider layers from locking variance slower due to signal dilution (where the average per-element gradient magnitude drops as width $W$ increases), we tried scaling gradients by their L2-norm (`raw_grad.abs()`) before locking variance.
*   **The Result:** Discarded.
*   **Why it was ruled out:** Empirical telemetry (`diagnostic_locking_v2.py`) showed this rescaled locking was an artificial intervention. It overrode the natural signal dilution properties of the architecture, forcing wider layers to lock as quickly as narrower ones. Slower locking in wider layers is an authentic emergent property of representation scaling, and forcing width-invariance obscured the true capacity characteristics of the system. We reverted `memory.py` to use raw unscaled gradients (`param.grad.data.abs()`).

---

## ❌ 2. Decoupled Boundary Loss
*   **The Idea:** Tested as an alternative loss function in the bottleneck layer to see if separating the boundaries of target facts would preserve sequential memories.
*   **The Result:** **0% recall (complete failure).**
*   **Why it was ruled out:** The mathematical structure of Decoupled Boundary Loss caused all downward gradients/forces to sum up and collapse all coordinates in the bottleneck layer to zero. 

---

## ❌ 3. Relative Margin Loss
*   **The Idea:** Tested as a margin-enforcing loss function to stabilize bottleneck coordinates.
*   **The Result:** **60% recall** (sub-optimal compared to standard Cross-Entropy).
*   **Why it was ruled out:** Relative Margin Loss generated unbalanced gradients that actively destroyed old templates to fit new coordinates. (Standard Cross-Entropy outperformed it at **80% recall** because its logit gradients sum to zero, creating a zero-sum force balance that stabilizes shared representation coordinates).

---

## ❌ 4. Sequential Online Classifier Training without Replay
*   **The Idea:** Training the classification readout layer sequentially using backpropagation (using either standard SGD or MESU) with zero replay.
*   **The Result:** **Catastrophic Forgetting & Primacy Bias (1.1% recall at N=100 facts, 0.12% at N=800).**
*   **Why it was ruled out:** The representation space is preserved (100% separable), but the classifier's sequential gradient updates shift decision boundaries globally. Metaplasticity locks parameter weights but cannot prevent the classification logits from shifting boundaries, rendering pure sequential training of a classification layer ineffective.

---

## ❌ 5. Pure Time-Based / Step-Based Variance Relaxation
*   **The Idea:** Using a constant step-based `alpha_decay` to unfreeze parameter variances (`var`) back to the prior.
*   **The Result:** **Stability-Plasticity Dilemma.**
*   **Why it was ruled out:** Constant time-based decay is blind to semantic novelty. If decay is too fast, the model forgets old templates; if it is too slow, the model saturates and cannot learn new facts. Prior relaxation must be neuromodulated (gated by novelty/loss) rather than occurring constantly.

---

## ❌ 6. Standard SGD + Sparse Replay (SGD Baseline)
*   **The Idea:** Relying on standard SGD optimizer combined with a small experience replay buffer (size 10).
*   **The Result:** **Recall collapse (2.0% recall).**
*   **Why it was ruled out:** SGD has no memory-protection mechanics. Even with a small replay buffer, it completely overwrites previous task templates. 

---

## ❌ 7. Calling `model.zero_grad()` before `engine.step()`
*   **The Idea:** Clearing gradients using native PyTorch `model.zero_grad()` immediately after the loss backward pass.
*   **The Result:** **Variance-locking telemetry completely broke.**
*   **Why it was ruled out:** Calling `model.zero_grad()` erases parameter gradients before the `engine.step()` has a chance to execute. This meant the MESU engine never saw gradient values, leaving parameter variance permanently locked at the prior ($\sigma^2_{\text{prior}}$). The correct pattern is to call `engine.step()` first, then clear gradients using `engine.zero_grad()`.

---

## ❌ 8. MESU as a Standalone Long-Horizon Optimizer
*   **The Idea:** Relying on weight-specific uncertainty tracking (MESU) alone without replay to solve sequential continual learning over long streams ($N \ge 50$ facts).
*   **The Result:** **Ruled out (only functions at small scale, e.g. $N \le 5$ facts).**
*   **Why it was ruled out:** 
    *   *Small Scale ($N=5$):* MESU successfully maintains **2.90/5 recall** (93.55% ratio of means) without replay, while SGD collapses to **0.20/5**.
    *   *Large Scale ($N \ge 50$):* Standalone recall decays to **0.12%** at 800 facts. In a fixed parameter network, capacity eventually saturates. If we lock variances, the model cannot learn new facts (plasticity saturation); if we unfreeze them via step-based prior relaxation, it overwrites old facts.

---

## ❌ 9. Scaling Layer Width under Unscaled Locking to Reduce Forgetting
*   **The Idea:** Increasing the bottleneck layer width (from 32 to 256) under the raw unscaled locking engine to increase memory capacity and reduce sequential forgetting.
*   **The Result:** **Ruled out (wider layers actually accelerate forgetting).**
*   **Why it was ruled out:** Wider layers distribute representations, making the raw gradient per weight smaller (signal dilution). Because the engine locks variance using unscaled gradients, this dilution causes wider layers to lock weight variances **slower**, keeping the parameters plastic and unprotected longer. As a result, wider networks are more vulnerable to sequential overwrites, yielding a steeper forgetting decay exponent ($\alpha_{256} = 1.2062$) compared to narrower ones ($\alpha_{64} = 1.0774$).
