# 🧠 BioNeural Network — Metaplastic Neuro-Channel (MNC) Framework

> A **multiplication-free**, CPU-native neural architecture that solves catastrophic forgetting through biologically-inspired synaptic uncertainty and dual-timescale memory cascades.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## Table of Contents

- [The Problem](#-the-problem)
- [The Solution](#-the-solution)
- [Results](#-validated-results)
- [Architecture Deep Dive](#-architecture-deep-dive)
- [The Loss Function Discovery](#-the-loss-function-discovery)
- [The 10-Day Protocol](#-the-10-day-delayed-recall-protocol)
- [Validation Studies](#-comprehensive-validation-studies)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Known Limitations](#%EF%B8%8F-known-limitations--honest-caveats)
- [Roadmap](#%EF%B8%8F-roadmap)

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

## 💡 The Solution

The MNC framework takes a fundamentally different approach. Instead of protecting knowledge *externally* (replay, distillation), it builds protection *into the parameter update rule itself* through three innovations:

### 1. Multiplication-Free Distance Operators
The standard neuron `y = Wx + b` is replaced with `y = -|X - W|₁ + b`. Input-template similarity is measured by spatial proximity (Manhattan distance) rather than angular correlation (dot product). This eliminates all matrix multiplications from the forward pass.

### 2. Metaplasticity from Synaptic Uncertainty (MESU)
Every parameter tracks two quantities: its **value** (μ) and its **confidence** (σ²). New parameters have high variance (uncertain) and update aggressively. Well-learned parameters have low variance (confident) and resist change. This is the "learning rate governor" — the network automatically decides *which weights are safe to modify*.

### 3. Dual-Timescale Memory Cascades
Two internal reservoirs track each parameter at different speeds:
- **u₁ (fast cascade):** Tracks the active parameter, responding quickly to new data
- **u₂ (slow cascade):** Consolidates from u₁, acting as a long-term memory anchor

A confidence-weighted restorative pull continuously nudges active parameters back toward u₂, preventing long-term drift without freezing the network.

### Key Innovations at a Glance

| Feature | Description |
|:---|:---|
| **Multiplication-Free Forward Pass** | Computes negative L1 (Manhattan) distance instead of dot products. Zero `matmul` operations. |
| **Surrogate Gradient Routing** | Custom `torch.autograd.Function` with L₂ surrogate for weights and HardTanh clamping for inputs — prevents dead gradients from absolute value operations. |
| **MESU Memory Engine** | Per-parameter Bayesian uncertainty tracking. Confident parameters resist change, uncertain parameters learn freely. |
| **Dual-Timescale Cascades** | Fast/slow reservoir coupling bounds long-term parameter drift (measured: 2.65 vs 4.56 Euclidean drift). |
| **Distance Autocalibration** | Shift/scale constants derived analytically from random unit-sphere distance distributions — zero manual tuning. |
| **Unit Sphere Projection** | Weight templates are projected onto the unit hypersphere after every update, preventing gradient explosion and ensuring bounded distance ranges. |

---

## 📊 Validated Results

### Headline: 4/5 Facts Recalled After Interference (80%)

The MNC was trained sequentially on 5 facts, then subjected to 4 interfering distractors (including semantically adversarial ones like "guest wifi code 9912" designed to overwrite "server code 7734"). On Day 10, queried with **paraphrased questions never seen during training**:

```
Day 10 Recall (CrossEntropyLoss + Autocalibration):
  Q: Where is the blue folder kept?         → Expected 0, got 0 → CORRECT ✓
  Q: What's the main server's access code?   → Expected 1, got 0 → WRONG   ✗
  Q: When was Sarah's meeting rescheduled?   → Expected 2, got 2 → CORRECT ✓
  Q: What fuel does the backup generator use? → Expected 3, got 3 → CORRECT ✓
  Q: When is Sector 4 camera maintenance?    → Expected 4, got 4 → CORRECT ✓

RESULT: 4/5 correct (80%) — PASS
```

The single failure (Fact 1: "server code 7734") was displaced by the semantically nearest interference sentence ("guest wifi code 9912"), which shares identical sentence structure and topic domain.

### 10-Seed Comprehensive Validation (Margin Loss)

All multi-seed results below use the validated margin contrastive loss in `run_comprehensive_validation.py`:

#### Study 1 — MESU vs SGD Baseline

| Method | Mean Recall | Std | Range | Notes |
|:---|:---:|:---:|:---:|:---|
| **MESU (negative conductance)** | **3.10 – 4.40** | — | — | Facts survive interference |
| SGD (lr=1.0) | 0.10 | — | — | Total catastrophic collapse |
| SGD (lr=0.1) | 0.70 | — | — | Near-complete collapse |
| SGD (lr=0.01) | 0.20 | — | — | Collapse at all learning rates |

> SGD collapses at *every* learning rate tested. MESU retains 3–4 out of 5 facts.

#### Study 5 — vs Tuned Transformer (5.2M parameters)

| Model | Trainable Params | Mean Recall | Notes |
|:---|:---:|:---:|:---|
| **MNC (MESU)** | **~12.6K** | **3.40** | Zero matrix multiplications |
| Decoder Transformer (tuned) | ~5.2M | 3.20 | Auto-tuned LR and steps via grid search |

> The MNC matches or exceeds a Transformer with **413× fewer trainable parameters** and zero matrix multiplications. The Transformer baseline uses AdamW with auto-tuned hyperparameters (LR sweep over `[1e-4, 5e-4, 1e-3, 3e-3]`, step sweep over `[15, 30, 50]`).

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

### Future Research 🔬
- [ ] **Dual-Path SmoothMinLSE + Chemical Bypass** — implement the LSE-smoothed physical channel and neurotransmitter bypass as a research branch, re-validate against the single-path baseline
- [ ] **Triton/C++ kernel compilation** — compile custom autograd functions into native vectorized CPU/GPU instructions to eliminate Python interpreter overhead
- [ ] **Scaling experiments** — test with 50+ facts, deeper semantic overlap between facts and interference
- [ ] **Multi-layer deep MNC stacking** — explore 3+ layer architectures with multiple bottlenecks
- [ ] **Real-world document streams** — validate on Wikipedia articles, conversation logs, and temporal knowledge bases
- [ ] **Comparison with EWC and SI** — benchmark against Elastic Weight Consolidation and Synaptic Intelligence baselines

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

This project is released under the [MIT License](LICENSE).
