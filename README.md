# 🧠 BioNeural Network — Metaplastic Neuro-Channel (MNC) Framework

> A **multiplication-free**, CPU-native neural architecture that solves catastrophic forgetting through biologically-inspired synaptic uncertainty and dual-timescale memory cascades.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 The Problem

Standard neural networks suffer from **catastrophic forgetting** — when trained sequentially on new information, they overwrite previously learned knowledge. Traditional mitigations (replay buffers, elastic weight consolidation) require storing historical data or computing expensive Fisher information matrices, both of which scale poorly for lifelong, on-device learning.

## 💡 The Solution

The MNC framework replaces standard matrix multiplication with **distance-based addition operators** and protects learned representations using **Metaplasticity from Synaptic Uncertainty (MESU)** — a Bayesian mechanism that tracks per-parameter confidence and automatically locks down well-learned weights while keeping uncertain parameters plastic.

### Key Innovations

| Feature | Description |
|:---|:---|
| **Multiplication-Free Forward Pass** | Computes negative L1 (Manhattan) distance instead of dot products. Zero `matmul` operations. |
| **Surrogate Gradient Routing** | Custom `torch.autograd.Function` with L₂ surrogate for weights and HardTanh clamping for inputs — prevents dead gradients from absolute value operations. |
| **MESU Memory Engine** | Every parameter tracks mean (μ) and variance (σ²). Update magnitude is proportional to uncertainty: confident parameters resist change, uncertain parameters learn freely. |
| **Dual-Timescale Cascades** | Fast execution variables (u₁) coupled to slow consolidation reservoirs (u₂). The slow cascade anchors parameters near consolidated coordinates, bounding long-term drift. |
| **Distance Autocalibration** | Shift/scale constants are derived analytically from the random unit-sphere distance distribution at initialization — no manual tuning required. |

---

## 📊 Validated Results (10-Seed Sweep)

All results are from the single-path `MNCAdderFunction` prototype, validated across 10 random seeds using the [10-Day Delayed Recall Protocol](plans/EVALUATION_PROTOCOL.md).

### Study 1 — MESU vs SGD Baseline

| Method | Mean Recall (out of 5) | Notes |
|:---|:---:|:---|
| **MESU (negative conductance)** | **3.10 – 4.40** | Retains facts through interference |
| SGD (lr=1.0) | 0.10 | Complete collapse |
| SGD (lr=0.1) | 0.70 | Near-complete collapse |

### Study 3 — Step Asymmetry Analysis

| Budget Configuration | MESU Recall |
|:---|:---:|
| Asymmetric (15 fact / 5 interference) | **3.10** |
| Symmetric Short (5 / 5) | **2.70** |
| Symmetric Long (15 / 15) | 1.80 |

### Study 4 — u₂ Cascade Drift Telemetry

| Configuration | Cascade Drift D(u₂, W₅) |
|:---|:---:|
| u₂ pull **enabled** | **2.65** |
| u₂ pull **disabled** | 4.56 |

### Study 5 — Transformer Control Baseline

| Model | Parameters | Mean Recall |
|:---|:---:|:---:|
| **MNC (MESU)** | **~12.6K** | **3.40** |
| Decoder Transformer (tuned) | ~5.2M | 3.20 |

> The MNC matches or exceeds a Transformer with **413× fewer parameters** and zero matrix multiplications.

---

## 🏗️ Architecture

```
Input Text
    │
    ▼
┌─────────────────────────┐
│  Frozen MiniLM Encoder  │   all-MiniLM-L6-v2 (CPU-only)
│  384-dim sentence embed │
└────────────┬────────────┘
             │ [1, 384]
             ▼
┌─────────────────────────┐
│   MNCLinear Layer 1     │   -|X - W|₁ summed (no matmul)
│   384 → 32 templates    │   + autocalibrated shift/scale
│   + Tanh activation     │
└────────────┬────────────┘
             │ [1, 32]
             ▼
┌─────────────────────────┐
│   MNCLinear Layer 2     │   -|X - W|₁ summed (no matmul)
│   32 → 10 outputs       │   + autocalibrated shift/scale
└────────────┬────────────┘
             │ [1, 10]
             ▼
       argmax → label
```

### The Custom Backward Pass

Standard absolute values produce flat sign-function gradients that kill learning. The MNC hijacks PyTorch's autograd with two surrogate routes:

- **Weight gradients (L₂ Surrogate):** `∂L/∂W ← (X − W)` — the full coordinate difference tells templates exactly how far and which direction to step.
- **Input gradients (HTDR):** `∂L/∂X ← clamp(W − X, -1, 1)` — HardTanh prevents chain-rule explosion through deeper layers.

### The MESU Memory Engine

```
For each parameter θ:
    σ²  ← uncertainty variance (initialized at σ²_prior)
    u₁  ← fast cascade (tracks active θ)
    u₂  ← slow cascade (consolidates from u₁)

On each step:
    1. Scale gradient adaptively: g̃ = √n / ||∇θ||₂ · ∇θ
    2. Apply uncertainty-gated update: θ -= lr · σ² · g̃
    3. Update cascades: u₁ += g·(θ - u₁),  u₂ += 0.1g·(u₁ - u₂)
    4. Pull θ toward u₂ (confidence-weighted): θ += conf · g · (u₂ - θ)
    5. Lock variance: σ² -= σ² · clamp(|g̃| · 0.2)
    6. Relax variance toward prior: σ² += α · (σ²_prior - σ²)
```

---

## 📁 Project Structure

```
bioneural-network/
├── mnc_project/
│   ├── mnc/
│   │   ├── kernels.py              # MNCAdderFunction — custom autograd primitive
│   │   ├── layers.py               # MNCLinear — multiplication-free layer module
│   │   └── memory.py               # MESUEngine — Bayesian continual learning optimizer
│   ├── data/
│   │   └── journal.txt             # 10-day synthetic journal dataset
│   ├── pipeline.py                 # Frozen MiniLM embedding pipeline (CPU)
│   ├── run_comprehensive_validation.py  # Full 10-seed, 5-study validation suite
│   ├── run_ablation_sweep.py       # MESU vs SGD ablation experiments
│   ├── run_recall_test.py          # Quick single-seed recall test
│   ├── run_audit.py                # Structural audit script
│   ├── test_alignment.py           # Embedding alignment verification
│   └── requirements.txt
├── plans/
│   ├── ARCHITECTURE.md             # Mathematical blueprint of the MNC primitive
│   ├── SYSTEM.md                   # Immutable coding constraints
│   ├── IMPLEMENTATION_PLAN.md      # Staged build plan with completion status
│   ├── EVALUATION_PROTOCOL.md      # 10-Day Delayed Recall test specification
│   └── ROADMAP.md                  # Research roadmap and risk register
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

### Run the Full Validation Suite

```bash
python run_comprehensive_validation.py
```

This executes all 5 studies across 10 random seeds:
- **Study 1:** MESU vs SGD with distance autocalibration
- **Study 2:** Hardened interference with shared output labels
- **Study 3:** Symmetric vs asymmetric step budget analysis
- **Study 4:** u₂ cascade drift telemetry
- **Study 5:** Tuned Decoder Transformer baseline comparison

### Run a Quick Recall Test

```bash
python run_recall_test.py
```

### Verify Individual Components

```bash
# Test the custom autograd kernel
python -m mnc.kernels

# Test the MNCLinear layer
python -m mnc.layers

# Test the MESU memory engine
python -m mnc.memory

# Test the embedding pipeline
python pipeline.py
```

---

## ⚠️ Known Limitations & Honest Caveats

1. **Small-Scale Proof of Concept.** The validation dataset is a synthetic 9-sentence journal (5 facts, 4 interference items). This demonstrates the mechanism works, not that it scales to production workloads.

2. **Step Budget Asymmetry Matters.** The network performs best when target facts receive more optimization steps than interference. Under symmetric long interference (equal budgets), recall degrades from ~3.1 to ~1.8.

3. **Single-Path Prototype.** The architectural blueprint describes a dual-path model (Physical Channel + Chemical Bypass with LSE smoothing). All validated results use the simpler single-path `MNCAdderFunction`. The dual-path design is deferred as a [future research milestone](plans/IMPLEMENTATION_PLAN.md).

4. **Embedding Dependency.** The MNC receives 384-dimensional embeddings from a frozen MiniLM encoder (~22M pretrained parameters). The comparison with the Transformer baseline is parameter-count-at-training-time, not total parameter count including the encoder.

---

## 🔬 How It Works — The 10-Day Protocol

```
Days 1–5:  Train sequentially on 5 target facts (batch_size=1, no replay)
           "The blue folder is in the third drawer."
           "The access code for the main server is 7734."
           ...

Days 6–9:  Train on interfering distractors designed to cause forgetting
           "The red folder is resting on the top desk."
           "The access code for the guest wifi is 9912."
           ...

Day 10:    Query the network on Day 1–5 facts (no further training)
           → How many of the 5 original facts does it still recall?
```

**SGD result:** 0.1/5 — total catastrophic forgetting.
**MESU result:** 3.1–4.4/5 — the uncertainty governor locks down learned templates.

---

## 🗺️ Roadmap

- [x] Custom autograd kernel (`MNCAdderFunction`)
- [x] MESU memory engine with dual-timescale cascades
- [x] Frozen MiniLM embedding pipeline
- [x] 10-seed comprehensive validation (Studies 1–5)
- [x] Tuned Transformer baseline comparison
- [x] Distance autocalibration
- [ ] Dual-path SmoothMinLSE + Chemical Bypass (research branch)
- [ ] Triton/C++ kernel compilation for native CPU vectorization
- [ ] Scaling to larger vocabularies and longer document streams
- [ ] Multi-layer deep MNC stacking experiments

---

## 📖 Citation

If you use this work in your research, please cite:

```
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
