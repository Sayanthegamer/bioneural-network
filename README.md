# 🧠 BioNeural Network — Metaplastic Neuro-Channel (MNC) Framework (A Negative Research Autopsy)

> **WARNING / POST-MORTEM:** This repository represents a completed negative research exploration of a "multiplication-free" neural architecture. While initially promising, rigorous mathematical and hardware-level stress testing has proven that the core design—specifically the L1 distance coordinate dispute, the one-way variance ratchet, and the embedding bottleneck—presents insurmountable limitations for scalable lifelong learning.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Table of Contents

- [The Autopsy: What Killed the Project](#-the-autopsy-what-killed-the-project-why-we-stopped)
- [The Problem](#-the-problem)
- [The Solution](#-the-solution)
- [Results](#-autopsy-results-the-bottleneck-shatters)
- [Architecture Deep Dive](#-architecture-deep-dive)
- [The Loss Function Discovery](#-the-loss-function-discovery)
- [The 10-Day Protocol](#-the-10-day-delayed-recall-protocol)
- [Validation Studies](#-comprehensive-validation-studies)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Known Limitations](#%EF%B8%8F-known-limitations--honest-caveats)
- [Roadmap](#%EF%B8%8F-roadmap)

---

## 💀 The Autopsy: What Killed the Project (Why We Stopped)

Following rigorous mathematical and hardware stress-testing under symmetric training, this architecture has been determined to be a **dead end**. It has been formally deprecated. 

Below is the autopsy of the structural and mathematical limits that killed the progress:

### 1. The Embedding Bottleneck (The Compute Lie)
The MNC claims to be a "multiplication-free" architecture designed to run on low-power edge CPUs. However, to compute semantic distances, the input data must first be converted into a dense, semantically aligned 384-dimensional vector. 
* **The Reality:** We must run a 22-million parameter Transformer (`all-MiniLM-L6-v2`) executing billions of standard matrix multiplications on the CPU *before the data ever reaches the MNC module*. 
* **The Verdict:** If you must fire up a deep neural network to embed every single incoming text packet, the gatekeeper is more computationally expensive than the gate itself. Bypassing the embedding model turns the MNC into a crude signal matcher, which is easily outclassed by standard, hyper-optimized classical algorithms like Locality Sensitive Hashing (LSH) or Isolation Forests.

### 2. The Variance Ratchet vs. Dynamic Tension
The optimizer was theoretically designed to dynamically unfreeze synaptic variance under persistent new evidence.
* **The Reality:** The update equation for variance $\sigma^2$ is a one-way subtractive ratchet: `var.sub_(var * torch.clamp(raw_grad.abs() * 0.2, max=0.25))`. High gradients from a new, related task do not unfreeze the variance; they lock the synapses down faster. The only mechanism that increases variance is a time-decay scalar (`alpha_decay`) completely blind to incoming data. 
* **The Verdict:** The system is an algorithmic lockbox that cannot dynamically adapt to new contexts without risking immediate catastrophic forgetting or absolute structural lock.

### 3. Zero-Sum Spatial Templates (No Forward Transfer)
To achieve general intelligence, a network must reuse features (Forward Transfer)—e.g., using a learned "circle" template to help learn a "sphere."
* **The Reality:** The MNC relies on L1 distances ($|X - W|$) which represent absolute spatial coordinates rather than multiplicative scaling. Moving a coordinate template closer to a new concept $B$ strictly increases its distance from the old concept $A$.
* **The Verdict:** Parameter sharing is a zero-sum territorial dispute. The network cannot abstract or reuse features; it can only occupy space or abandon it.

### 4. The VRAM / Parameter Trap
To track memory without a replay buffer, the MESU engine tracks the parameter weight, variance, and dual-timescale cascades ($u_1$, $u_2$).
* **The Reality:** This quadruples the memory footprint of every parameter. Scaled to a standard 100M parameter model, the optimizer state alone balloons to 3x the model size (400M params).
* **The Verdict:** For that exact same VRAM budget, a standard model using a bounded **Experience Replay Buffer** (reservoir sampling) achieves mathematically superior recall with $O(1)$ scaling and zero parameter bloat.

### 5. Jagged Manifolds & Twitched Actuators
* **The Reality:** L1 distance operators and `HardTanh` gradient clamping produce a piecewise-linear manifold riddled with sharp corners and dead zones.
* **The Verdict:** While viable for toy classification, this jagged landscape is physically incompatible with continuous control systems (like robotics), where discontinuous gradient jumps manifest as violent, destructive physical actuator twitch.

---

## 🎯 The Problem

Standard neural networks suffer from **catastrophic forgetting** — when trained sequentially on new information, they overwrite previously learned knowledge. This is the fundamental barrier to lifelong learning on edge devices.

Traditional mitigations all have costs:

| Method | Limitation |
|:---|:---|
| **Replay Buffers** | Requires storing historical data — scales linearly with experience |
| **Elastic Weight Consolidation (EWC)** | Requires computing the Fisher Information Matrix — O(n²) per parameter |
| **Progressive Neural Networks** | Requires growing the network — unbounded memory |
| **Knowledge Distillation** | Requires a separate teacher model — doubles compute |

All of these assume access to historical data or expensive second-order statistics. None work for truly online, single-pass, on-device learning.

## 💡 The Hypothesized Solution & Why It Failed

The MNC framework hypothesized that instead of protecting knowledge *externally* (replay, distillation), it could build protection *into the parameter update rule itself* through three innovations. 

Here is the breakdown of the hypothesized mechanics and why they mathematically failed during stress testing:

### 1. Multiplication-Free Distance Operators (Falsified)
* **Hypothesis:** Replace standard dot-product neurons with `y = -|X - W|₁ + b`. Measuring similarity via spatial proximity (Manhattan distance) eliminates matrix multiplications.
* **Why it Failed:** Processing continuous semantic inputs still requires generating the embedding $X$ first, which relies on a standard, multiplication-heavy 22M parameter Transformer. Bypassing the Transformer turns the MNC into a simple signal matcher, which is slower and less accurate than classical LSH or Bloom filters. Additionally, L1 distance coordinates turn updates into a zero-sum territorial dispute, preventing feature reuse (no forward transfer).

### 2. Metaplasticity from Synaptic Uncertainty (MESU) (Falsified)
* **Hypothesis:** Track parameter values ($\mu$) and confidence ($\sigma^2$) to act as a dynamic, uncertainty-scaled learning rate governor.
* **Why it Failed:** The update equations act as a one-way subtractive ratchet. Under gradient flow, variance only shrinks, permanently freezing weights. There is no mechanism to unfreeze variances based on new evidence, causing catastrophic remembering.

### 3. Dual-Timescale Memory Cascades (Falsified)
* **Hypothesis:** Couple fast-timescale reservoirs ($u_1$) and slow long-term anchors ($u_2$) to prevent parameter drift.
* **Why it Failed:** The extra tracking variables quadruple the parameter memory footprint. This $O(P)$ VRAM premium is far less efficient than a simple, bounded Experience Replay Buffer that achieves mathematically superior recall with $O(1)$ memory.

### Hypothesized Mechanics at a Glance

| Feature | Intended Concept | Autopsy Verdict / Failure Mode |
|:---|:---|:---|
| **Multiplication-Free Pass** | Computes negative L1 distance instead of dot products. | **Falsified:** Embedding extraction requires standard multiplication-heavy Transformers; L1 metrics cause zero-sum coordinate conflicts. |
| **Surrogate Gradient Routing** | Custom autograd with L₂ weight surrogate and HardTanh input clamping. | **Falsified:** Piecewise-linear L1 manifolds create jagged gradients and dead zones, causing actuator twitch in control loops. |
| **MESU Memory Engine** | Per-parameter Bayesian uncertainty scaling. | **Falsified:** Update rules function as a one-way variance ratchet ($\sigma^2 \to 0$), causing rigid lockup. |
| **Dual-Timescale Cascades** | Fast/slow reservoir coupling bounds parameter drift. | **Falsified:** Tracking four states per weight quadruples memory; a 1.2GB replay buffer offers better recall at lower VRAM cost. |

---

## 📊 Autopsy Results: The Bottleneck Shatters

When subjected to a realistic, **symmetric training protocol** (where new information and old information are trained with equal step budgets), the architecture collapses, failing to achieve stable continual learning.

### Symmetric Recall Test Result (run_recall_test.py):
```
Day 10 Recall (Symmetric CrossEntropyLoss + Autocalibration):
  Q: Where is the blue folder kept?         → Expected 0, got 0 → CORRECT ✓
  Q: What's the main server's access code?   → Expected 1, got 7 → WRONG   ✗
  Q: When was Sarah's meeting rescheduled?   → Expected 2, got 7 → WRONG   ✗
  Q: What fuel does the backup generator use? → Expected 3, got 3 → CORRECT ✓
  Q: When is Sector 4 camera maintenance?    → Expected 4, got 4 → CORRECT ✓

RESULT: 3/5 correct (60%) — FAIL
```

The system fails because MESU operates as a **one-way variance ratchet** ($\sigma^2 \to 0$), freezing parameter values rather than unfreezing them dynamically based on semantic relation. Additionally, the $L_1$ distance operator acts as an absolute coordinate template, turning spatial parameter updates into a zero-sum territorial dispute that actively prevents forward knowledge transfer.

---

## 🏗️ Architecture Deep Dive

### Network Topology

```
Input Text
    │
    ▼
┌──────────────────────────────┐
│    Frozen MiniLM Encoder     │   all-MiniLM-L6-v2 (22M params, CPU-only)
│    384-dim sentence embed    │   Not trained — frozen feature extractor
└─────────────┬────────────────┘
              │ [1, 384]
              ▼
┌──────────────────────────────┐
│    MNCLinear Layer 1         │   384 input features → 32 templates
│    y = -|X - W|₁             │   Negative L1 distance (no matmul)
├──────────────────────────────┤
│    ScaleDistances            │   Autocalibrated: shift=22.09, scale=1.22
│    y = (y + shift) / scale   │   Centers distribution near 0 for Tanh
├──────────────────────────────┤
│    Tanh Activation           │   Squashes to [-1, 1], creates nonlinearity
└─────────────┬────────────────┘
              │ [1, 32]       ← THE BOTTLENECK (shared representation)
              ▼
┌──────────────────────────────┐
│    MNCLinear Layer 2         │   32 input features → 10 output classes
│    y = -|X - W|₁             │   Each output node is a "memory template"
├──────────────────────────────┤
│    ScaleDistances            │   Autocalibrated: shift=6.39, scale=1.31
└─────────────┬────────────────┘
              │ [1, 10]
              ▼
        argmax → predicted label
```

### The Forward Primitive: `MNCAdderFunction`

The core computation replaces dot products with distance measurement:

```python
# Standard neuron (multiplication-based):
y = torch.matmul(x, W.T) + b      # O(n²) multiplications

# MNC neuron (addition-based):
diff = x.unsqueeze(1) - w.unsqueeze(0)   # Pairwise differences
y = -torch.abs(diff).sum(dim=2)          # Negative L1 distance (additions only)
```

**Why negative L1?** The closest template to the input produces the *least negative* output (closest to 0). `argmax` naturally selects the best-matching template.

### The Custom Backward Pass: Surrogate Gradient Routing

The true derivative of `|x|` is the `sign` function — a flat step that provides zero useful gradient information. The MNC hijacks PyTorch's autograd chain with mathematically motivated surrogates:

```python
@staticmethod
def backward(ctx, grad_output):
    diff, = ctx.saved_tensors

    # Weight gradients — L₂ Surrogate
    # Instead of sign(X-W), pass the full difference (X-W).
    # This tells each template exactly how far and which direction
    # to step to catch the input. Smooth, continuous, no dead zones.
    grad_w = (grad_output.unsqueeze(2) * diff).sum(dim=0)

    # Input gradients — HardTanh Derivative Replacement (HTDR)
    # Clamp to [-1, 1] to prevent chain-rule explosion when
    # gradients propagate backward through multiple layers.
    grad_x_diff = torch.clamp(diff, min=-1.0, max=1.0)
    grad_x = -(grad_output.unsqueeze(2) * grad_x_diff).sum(dim=1)

    return grad_x, grad_w
```

### The MESU Memory Engine

The optimizer that makes continual learning work:

```
INITIALIZATION:
    For each parameter θ in the model:
        σ²[θ]  ← σ²_prior              (uncertainty starts high — "I know nothing")
        u₁[θ]  ← copy of θ             (fast cascade initialized to weights)
        u₂[θ]  ← copy of θ             (slow cascade initialized to weights)

ON EACH TRAINING STEP:
    1. ADAPTIVE GRADIENT SCALING
       g̃ = (√num_params / ||∇θ||₂) · ∇θ
       ↳ Normalizes gradient magnitude per-layer to prevent scale mismatch

    2. UNCERTAINTY-GATED UPDATE (The Learning Rate Governor)
       θ  -=  lr × σ² × g̃
       ↳ High σ² (uncertain) → large update (learn aggressively)
       ↳ Low σ² (confident)  → tiny update (protect this memory)

    3. DUAL-TIMESCALE CASCADE UPDATE
       g = 0.1 × sigmoid(-current_loss)         (coupling conductance)
       u₁ += g × (θ - u₁)                       (fast tracks active params)
       u₂ += 0.1g × (u₁ - u₂)                  (slow consolidates from fast)

    4. RESTORATIVE PULL (Memory Anchoring)
       confidence = clamp(1 - σ²/σ²_prior, 0, 1)
       θ  +=  confidence × g × (u₂ - θ)
       ↳ Pulls confident parameters back toward their consolidated state
       ↳ Measured effect: reduces cascade drift from 4.56 to 2.65

    5. VARIANCE LOCKING
       σ² -= σ² × clamp(|g̃| × 0.2, max=0.25)
       ↳ Large gradients = "this param changed a lot" → lock it down

    6. PRIOR RELAXATION (Anti-Freezing)
       σ² += α_decay × (σ²_prior - σ²)
       ↳ Prevents permanent freezing (catastrophic remembering)
       ↳ Slowly reopens plasticity for parameters that haven't been used

    7. UNIT SPHERE PROJECTION
       W = W / ||W||₂  (per-row normalization)
       ↳ Keeps all templates on the unit hypersphere
       ↳ Bounds distance ranges, prevents gradient explosion
```

### Distance Autocalibration

The ScaleDistances layers require knowing the expected distribution of raw L1 distances between random unit-sphere vectors. Instead of hand-tuning constants, we derive them analytically:

```python
def autocalibrate_scale_distances(in_features, num_samples=1000):
    # Sample random unit-sphere vectors
    x = random_unit_vectors(num_samples, in_features)
    w = random_unit_vectors(num_samples, in_features)

    # Compute expected negative L1 distance distribution
    neg_l1 = -torch.norm(x - w, p=1, dim=1)

    # Center the distribution at 0, scale to unit variance
    shift = -neg_l1.mean()    # ≈ 22.09 for 384-dim
    scale = 2.0 * neg_l1.std()  # ≈ 1.22 for 384-dim

    return shift, scale
```

This ensures that the Tanh activation receives inputs centered near 0 with std ≈ 0.5 — preventing saturation that kills gradient flow to deeper layers.

---

## 🔑 The Loss Function Discovery

One of the most critical findings during development: **the choice of loss function has a dramatic effect on bottleneck architectures**, and the "obvious" choices can catastrophically fail.

### The Bottleneck Problem

In our architecture, Layer 0 (384→32) has **shared weights** used by all 10 output classes. Any gradient that flows backward through the network creates forces on these shared weights. If those forces are unbalanced, the shared layer collapses.

### Three Loss Functions Tested

#### 1. Relative Margin Loss (60% recall)
```python
# For each wrong class, penalize if it's within margin of the correct class
loss += clamp(logits[wrong] - logits[correct] + margin, min=0)
```
**Problem:** When a new, untrained class has logit = -14.0 and an old fact has logit = -0.5, the loss is `(-0.5) - (-14.0) + 1.0 = 14.5`. This massive gradient simultaneously pulls the new class up AND shoves the old template down. The old memory is actively destroyed to make room.

**Result: 3/5 correct (60%)**

#### 2. Decoupled Boundary Loss (0% recall)
```python
# Pull correct class up to absolute safe zone
loss = clamp(-logits[correct] - 0.2, min=0)
# Push wrong classes below absolute boundary
loss += sum(clamp(logits[wrong] + 1.0, min=0))
```
**Problem:** This fires for ALL 9 wrong classes simultaneously, regardless of whether they're actually competing. In a bottleneck, 9 classes push Layer 0 **down** while only 1 pulls it **up**. Net force: 8 units of destruction. The shared layer collapses immediately.

**Result: 0/5 correct (0%) — catastrophic**

#### 3. CrossEntropyLoss (80% recall) ✅
```python
loss = CrossEntropyLoss(logits, target)
```
**Why it works:** CrossEntropyLoss has a mathematical property that makes it uniquely suited for bottleneck architectures — its **gradients on logits always sum to exactly zero**:

```
∂L/∂z_j = p_j - y_j

where p_j = softmax(z_j) and y_j = 1 if j is the target, 0 otherwise.

Sum: ∑(p_j - y_j) = ∑p_j - ∑y_j = 1 - 1 = 0
```

This means the shared bottleneck layer receives **perfectly balanced forces** — the pull-up from the correct class exactly cancels the push-down from all wrong classes. The bottleneck stays stable while per-class templates in Layer 2 learn independently.

**Result: 4/5 correct (80%) — PASS** ✅

| Loss Function | Bottleneck Force Balance | Recall |
|:---|:---|:---:|
| Relative Margin | Partial coupling — partially destructive | 60% |
| Decoupled Boundary | 9-vs-1 imbalance — maximally destructive | **0%** |
| **CrossEntropyLoss** | **Zero-sum — perfectly balanced** | **80%** ✅ |

---

## 🔬 The 10-Day Delayed Recall Protocol

The evaluation simulates a realistic sequential learning scenario where a system must retain early knowledge through later interference:

```
┌─────────────────────────────────────────────────────────────┐
│ Days 1–5: CONSOLIDATED CONTEXT (Target Facts)               │
│                                                             │
│   Day 1: "The blue folder is in the third drawer."          │
│   Day 2: "The access code for the main server is 7734."     │
│   Day 3: "Sarah's meeting was moved to Tuesday at 3 PM."    │
│   Day 4: "The backup generator requires unleaded fuel."     │
│   Day 5: "Sector 4 cameras undergo maintenance at midnight."│
│                                                             │
│   Training: 15 gradient steps per fact, batch_size=1        │
│   Noise injection: Gaussian ε~N(0, 0.05) per step          │
├─────────────────────────────────────────────────────────────┤
│ Days 6–9: INTERFERING CONTEXT (Designed to Cause Forgetting)│
│                                                             │
│   Day 6: "The red folder is resting on the top desk."       │
│   Day 7: "Someone left a coffee mug in the breakroom."      │
│   Day 8: "The access code for the guest wifi is 9912."      │
│   Day 9: "It is raining heavily outside today."             │
│                                                             │
│   Training: 3 gradient steps per distractor, batch_size=1   │
│   Note: Day 8 is adversarial — semantically attacks Day 2   │
├─────────────────────────────────────────────────────────────┤
│ Day 10: QUERY (No Training — Pure Recall)                   │
│                                                             │
│   Queried with PARAPHRASED questions:                       │
│   "Where is the blue folder kept?"         (not "...third   │
│   "What's the main server's access code?"    drawer")       │
│   ...etc.                                                   │
│                                                             │
│   The network must generalize from training sentences to    │
│   semantically equivalent but lexically different queries.  │
└─────────────────────────────────────────────────────────────┘
```

### Strict Constraints

- **Batch size = 1** — no mini-batching, no DataLoaders
- **No replay buffer** — once a sentence is trained on, its tensor is deleted
- **No historical storage** — the network never sees a previous sentence again
- **Paraphrased queries** — Day 10 questions are semantically equivalent but use different wording than training sentences

---

## 🧪 Comprehensive Validation Studies

The full validation suite (`run_comprehensive_validation.py`) runs 5 studies across 10 random seeds:

### Study 1: Baseline MESU vs SGD Sweep
Tests whether MESU's uncertainty-gating provides genuine protection against sequential forgetting, or whether simple SGD with the right learning rate could achieve the same result.

**Finding:** SGD collapses at every learning rate (0.0–0.7/5). MESU retains 3.1–4.4/5. The gap is real and large.

### Study 2: Hardened Interference (Shared Output Labels)
Forces interference sentences to share the same output label slots as target facts (e.g., "red folder" maps to label 0, same as "blue folder"), creating direct representational conflict.

**Finding:** MESU retains facts with only 0.50 average displaced facts vs total SGD collapse.

### Study 3: Step Budget Symmetry Analysis
Tests whether the recall advantage comes from MESU or from giving facts more training steps than interference (15 vs 3).

| Config | Fact Steps | Interference Steps | MESU Recall |
|:---|:---:|:---:|:---:|
| Asymmetric | 15 | 3 | **3.10** |
| Symmetric Short | 5 | 5 | **2.70** |
| Symmetric Long | 15 | 15 | 1.80 |

**Finding:** Asymmetry helps, but MESU retains an advantage even under symmetric budgets (2.70 vs SGD collapse). Under long symmetric interference (15/15), recall degrades — MESU is necessary but not sufficient alone.

### Study 4: u₂ Cascade Drift Telemetry
Measures whether the slow cascade (u₂) restorative pull actually bounds parameter drift, or whether it's decorative.

| Metric | u₂ Enabled | u₂ Disabled |
|:---|:---:|:---:|
| Cascade Drift D(u₂, W₅) | **2.65** | 4.56 |
| Parameter Drift D(W₁₀, W₅) | lower | higher |

**Finding:** The restorative pull measurably reduces drift (42% reduction). Its impact on short-horizon recall is modest, but it's critical for long-term stability.

### Study 5: Transformer Control Baseline
Compares against a properly tuned 5.2M parameter Decoder-Only Transformer with:
- 6-layer, 8-head architecture (d_model=256, FFN=1120)
- AdamW optimizer with weight decay
- Auto-tuned hyperparameters (LR and steps grid search)
- Same sequential, unbatched training protocol

**Finding:** The 12.6K parameter MNC matches or exceeds the 5.2M parameter Transformer (3.40 vs 3.20 mean recall).

---

## 📁 Project Structure

```
bioneural-network/
├── mnc_project/
│   ├── mnc/                            # Core framework
│   │   ├── kernels.py                  # MNCAdderFunction — custom autograd primitive
│   │   │                               #   Forward: -|X-W|₁ (negative L1 distance)
│   │   │                               #   Backward: L₂ surrogate (W), HTDR (X)
│   │   ├── layers.py                   # MNCLinear — multiplication-free nn.Module
│   │   │                               #   Unit sphere initialization + bias
│   │   └── memory.py                   # MESUEngine — Bayesian continual learning optimizer
│   │                                   #   Variance tracking, cascade coupling, prior relaxation
│   ├── data/
│   │   └── journal.txt                 # 10-day synthetic journal (5 facts + 4 interference)
│   ├── pipeline.py                     # JournalPipeline — frozen MiniLM embedding (CPU)
│   │                                   #   embed_sentence() → [1, 384]
│   │                                   #   embed_tokens()   → [1, Seq_Len, 384]
│   ├── run_comprehensive_validation.py # Full 10-seed, 5-study validation suite
│   │                                   #   Studies 1-5 with telemetry and baselines
│   ├── run_ablation_sweep.py           # MESU vs SGD ablation across 10 seeds
│   ├── run_recall_test.py              # Quick single-seed recall test (CrossEntropyLoss)
│   ├── run_audit.py                    # Structural integrity audit
│   ├── test_alignment.py               # Embedding alignment verification
│   └── requirements.txt               # torch, sentence-transformers, numpy, psutil
├── plans/
│   ├── ARCHITECTURE.md                 # Mathematical blueprint of the MNC primitive
│   ├── SYSTEM.md                       # Immutable coding constraints (no matmul, no batching)
│   ├── IMPLEMENTATION_PLAN.md          # Staged build plan (Stages 1-4 complete, 5 deferred)
│   ├── EVALUATION_PROTOCOL.md          # 10-Day Delayed Recall Protocol specification
│   └── ROADMAP.md                      # Research roadmap, risk register, open focus areas
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PyTorch 2.1+
- ~80MB disk space for the MiniLM model (auto-downloaded on first run)

### Installation

```bash
git clone https://github.com/Sayanthegamer/bioneural-network.git
cd bioneural-network/mnc_project

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Quick Recall Test (Recommended First Run)

```bash
python run_recall_test.py
```

Expected output: `4/5 correct (80%) — PASS`

### Run the Full 10-Seed Validation Suite

```bash
python run_comprehensive_validation.py
```

This executes all 5 studies across 10 random seeds (~5-10 minutes on CPU):
- **Study 1:** MESU vs SGD baseline sweep with autocalibration
- **Study 2:** Hardened interference with shared output labels
- **Study 3:** Symmetric vs asymmetric step budget analysis
- **Study 4:** u₂ cascade drift telemetry and ablation
- **Study 5:** Tuned Decoder Transformer baseline comparison

### Run the Ablation Sweep

```bash
python run_ablation_sweep.py
```

### Verify Individual Components

```bash
# Test the custom autograd kernel (forward + backward pass)
python -m mnc.kernels

# Test the MNCLinear layer (gradient flow through nn.Module)
python -m mnc.layers

# Test the MESU memory engine (variance locking + cascade coupling)
python -m mnc.memory

# Test the embedding pipeline (sentence → [1, 384] tensor)
python pipeline.py
```

---

## ⚠️ Known Limitations & Honest Caveats

These are real limitations that anyone reproducing or building on this work should understand:

1. **Small-Scale Proof of Concept.** The validation dataset is a synthetic 9-sentence journal (5 facts, 4 interference items). This demonstrates the mechanism works at small scale. Scaling to hundreds of facts with deep semantic overlap remains unverified.

2. **Step Budget Asymmetry is a Necessary Safeguard.** The network performs best when target facts receive more optimization steps (15) than interference (3). Under symmetric long interference (15/15), recall degrades to ~1.80/5. The MESU governor is necessary but not sufficient alone against sustained, equal-budget interference.

3. **Single-Path Prototype.** The [architectural blueprint](plans/ARCHITECTURE.md) describes a dual-path model (Physical Channel + Chemical Bypass with LSE smoothing). All validated results use the simpler single-path `MNCAdderFunction`. The dual-path design is deferred as a [future research milestone](plans/IMPLEMENTATION_PLAN.md) because changing the gradient landscape could materially alter MESU consolidation dynamics.

4. **Embedding Dependency.** The MNC receives 384-dimensional embeddings from a frozen `all-MiniLM-L6-v2` encoder (~22M pretrained parameters). The Transformer comparison is parameter-count-at-training-time (12.6K vs 5.2M), not total parameter count including the frozen encoder. Both the MNC and the Transformer baseline receive MiniLM embeddings as input, making the comparison fair at the training layer.

5. **Layer 0 Gradient Bottleneck.** The HTDR clamping in the backward pass limits gradient magnitude flowing to Layer 0. This means the 384→32 bottleneck weights move slowly (gradient magnitude ~0.00005 vs Layer 2's ~0.14). The network compensates by learning most of its discrimination in Layer 2's per-class templates, with Layer 0 providing a coarse shared representation. This is an architectural constraint, not a bug, but it limits the bottleneck's representational capacity.

6. **Single-Seed Transformer Tuning Variance.** The Transformer baseline's auto-tuning sweep is performed on Seed 0 only. The optimal hyperparameters found may not generalize equally across all seeds, contributing to variance in the Transformer's cross-seed scores.

7. **The Capacity Wall:** A 12.6K parameter network violates information theory constraints for lifelong episodic storage. As the network saturates, it is mathematically forced into either catastrophic forgetting (relaxing priors) or catastrophic remembering (variance lockout).
8. **The Discontinuous Gradient Trap:** L1 norms and HTDR clamping generate jagged, piecewise-linear manifolds. This prevents the smooth generative synthesis required for offline replay and causes pipeline stalls on modern MAC-optimized hardware.
9. **The Generative Replay Paradox:** The architecture survives its bottleneck via lossy compression. Generating training data from these degraded centroids to fine-tune the Transformer during "sleep" replay will poison the Transformer's precision manifolds.

---

## 🗺️ Roadmap

### Completed ✅
- [x] Custom autograd kernel — `MNCAdderFunction` with L₂/HTDR surrogate gradients
- [x] MESU memory engine — variance tracking, cascade coupling, prior relaxation
- [x] Frozen MiniLM embedding pipeline — CPU-only, sequential, no replay
- [x] Distance autocalibration — analytically derived from sphere distance distributions
- [x] CrossEntropyLoss integration — zero-sum gradients for bottleneck stability
- [x] 10-seed comprehensive validation — Studies 1–5 with full telemetry
- [x] Tuned Transformer baseline comparison — auto-tuned 5.2M param decoder
- [x] Unit sphere weight projection — bounds template norms after every update

---

## 🧬 Biological Inspiration

The MNC framework draws from real neuroscience:

| Biological Mechanism | MNC Implementation |
|:---|:---|
| **Synaptic metaplasticity** | MESU variance-gated updates — synapses that have been recently strengthened resist further modification |
| **Hebbian consolidation** | u₂ slow cascade — memories are gradually consolidated from short-term (u₁) to long-term (u₂) storage |
| **Homeostatic plasticity** | Prior relaxation — prevents permanent synaptic silencing by slowly restoring baseline plasticity |
| **Dendritic distance computation** | L1 distance operator — biological neurons compute similarity through spatial proximity of dendritic inputs, not multiplicative correlation |

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{mnc_framework_2026,
  author = {Sayanthegamer},
  title = {BioNeural Network: Multiplication-Free Continual Learning via Metaplastic Synaptic Uncertainty},
  year = {2026},
  url = {https://github.com/Sayanthegamer/bioneural-network}
}
```

---

## 📄 License

This project is released under the [Apache 2.0 License](LICENSE).
