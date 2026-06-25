# 🧠 BioNeural Network — Metaplastic Neuro-Channel (MNC) Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Bounded Continual Learning via Metaplastic Synaptic Uncertainty: Results, Limitations, and Open Questions**

## 🎯 The Vision: Three Core Pillars
The Metaplastic Neuro-Channel (MNC) is a research-grade neural architecture designed from the ground up to solve three fundamental challenges in modern AI:

1. **Resolving Catastrophic Forgetting:** Neural networks must learn sequentially over a lifetime without a replay buffer destroying old knowledge to make room for new data.
2. **Budget Hardware Accessibility:** AI must run on mobile devices and budget laptops without requiring massive, expensive GPU clusters.
3. **Extreme Energy Efficiency:** Like the human brain, the architecture must operate efficiently, minimizing power consumption through fundamentally cheaper mathematical operations.

---

## ⚙️ Architectural Design: How We Achieve the Goals

To meet these goals, the MNC discards standard Cross-Correlation (Matrix Multiplication) entirely. It replaces the computational bottlenecks of standard Transformers with ALU-friendly, multiplication-free flow control operations.

| Component / Feature | Mechanism for Efficiency & Continual Learning |
| :--- | :--- |
| **L1 Spatial Distance** | The core forward pass uses $y = -\|x - w\|_1 + b$. By bypassing `matmul`, the network drastically reduces computational cost and energy usage. |
| **Custom Autograd Kernels** | To prevent dead gradients from hard clamping, the backward pass uses a decoupled $L_2$ weight surrogate and HardTanh input clamping, implemented in custom PyTorch/Triton kernels. |
| **The MESU Engine** | **M**etaplasticity from **S**ynaptic **U**ncertainty. Every parameter tracks its Bayesian variance ($\sigma^2$). Update magnitudes are gated by uncertainty: confident memories are locked down to prevent catastrophic forgetting, while uncertain parameters remain plastic. |
| **Dual-Timescale Cascades** | A slow-moving cascade ($u_2$) pulls active parameters back toward consolidated coordinates, anchoring templates and bounding long-term parameter drift. |

---

## 🔬 Early Validation & Diagnostics (Phase 1)

Before stress-testing at scale, the core framework was validated against a 10-Day Delayed Recall Protocol (Sequential training on Days 1-5, Interference on Days 6-9, Query on Day 10).

**Key Findings from Initial Diagnostics:**
*   **MESU vs. SGD Baseline:** Standard SGD collapses (0.20/5 recall) under sequential interference. The MESU engine maintains **2.90/5 recall** (93.5% preservation ratio).
*   **Orthogonal Gradients:** Cosine similarity of task updates is near-zero ($\approx -0.02$ in Layer 2), disproving the assumption that sequential tasks uniformly compete for parameter coordinates.
*   **Variance Ratchet Resolution:** A time-based prior relaxation mechanism (`alpha_decay`) ensures variance does not permanently collapse to zero, maintaining a healthy plasticity equilibrium.
*   **Parameter Efficiency:** A 12.6K parameter MNC bottleneck achieves comparable sequential recall to a 5.2M parameter baseline Transformer on the same data stream.

---

## 🧪 The Scientific Audits: Discovering the Capacity Ceilings (Phase 2)

To understand the true scaling laws of the architecture, we conducted a rigorous 5-part experimental suite (Experiments 1 through 5). This journey revealed that the bottleneck to lifelong learning isn't just parameter overwriting—it is the geometry of the semantic space itself.

### 📊 Experiment 1: Width-Scaling & The Capacity Wall
* **Objective:** Determine if simply making the bottleneck wider ($W=32 \to W=512$) solves forgetting.
* **Finding:** We hit a severe capacity wall. Without replay, recall decays following a power-law $1/N$ ($48\%$ at $N=5$, dropping to $1.1\%$ at $N=100$). Crucially, **forgetting is interference-dominated**. We confirmed a universal geometric density collapse principle: crowding is proportional to $N/W$. Widening the network beyond $W \ge 128$ yielded zero benefit if the underlying embedding manifold lacked the necessary separation space.

### 📊 Experiment 2: Breaking the Prototype Assumption
* **Objective:** Does averaging statement embeddings into a single class centroid (Prototype) destroy metric details compared to instance-based K-Nearest Neighbors (k-NN)?
* **Finding:** We mathematically falsified the centroid compression bottleneck. Prototype classification performed nearly identically to instance-based 1-NN lookup (e.g., 92.6% vs 90.0% at N=800). The prototype readout head design is optimal; the true bottleneck lies in projection distortion and manifold boundaries.

### 📊 Experiment 3 & 4: Relational Aliasing & The "28% Wall"
* **Objective:** Audit the memory capacity using structured relational facts (e.g., "The blue folder for Project X is in Room Y").
* **Finding:** In Experiment 3, recall completely flatlined at $\sim 28\%$ even at small $N$. We discovered **Relational Aliasing**. Due to the combinatorial limits of the synthetic generator, an average of 3.56 distinct facts shared the *exact same query text*. This forced an absolute **Theoretical Bayes Ceiling** of $1/3.56 \approx 28.1\%$.
* **The Reranker Verification (Exp 4):** A secondary cross-encoder reranker yielded 0.0% gain. However, the target fact was found in the Top-10 retrieved candidates 100% of the time (`Coverage@10 = 100%`). This proved the memory was functioning perfectly; the failure was purely due to unresolvable query ambiguity.

### 📊 Experiment 5: Unambiguous Schema Scaling & True Capacity ($N^*$)
* **Objective:** Having isolated the aliasing flaw, we generated an unambiguously aliased dataset (max_alias = 1) scaling up to $N=3200$ to find the true representation breakpoint $N^*$ (where Recall@1 $\ge 95\%$ and P5 Margin $\ge 0$).
* **Finding:**
  * The raw uncompressed 384D space supports $N^* = 200$ perfectly retrieved sequential facts.
  * However, under a **Random linear projection** to a $W=128$ bottleneck, $N^*$ drops below 100 (Recall falls to 85%).
  * By applying an **Oracle-SVD** projection matrix (fitting the specific data manifold), we restored $W=128$ performance to 94.5% at $N=400$, proving that dimensionality compression is only viable if the projection is deeply aligned with the data manifold structure.

*(Note: Oracle-SVD represents a theoretical upper-bound ceiling, as computing it relies on the same statement corpus being retrieved—a form of data leakage. It illustrates what is geometrically possible, not a current deployment reality).*

---

## 📓 Verified Limitations & Ruled-Out Decisions

Our aggressive falsification process ruled out several assumptions:
1. **SGD + Sparse Replay Fails:** Using a small replay buffer (size 10) with standard SGD collapses to 2.0% recall.
2. **Sequential Online Classifiers Fail:** Updating the classification layer sequentially without replay causes boundaries to shift globally, yielding a 1.1% recall at N=100.
3. **Larger Encoders are Not Always Better:** 768-dimensional encoders (MPNet, E5-base) actually performed *worse* than the 384D `MiniLM` under random projection because sparser high-dimensional manifolds suffer from higher distortion.

---

## 🚀 Future Roadmap

With the MESU v1 framework thoroughly mapped, the next phase focuses on overcoming the identified projection distortion bottlenecks:
1. **Learned SVD / Data-Aligned Projections:** Replacing the random linear bottleneck projections with learned weights that minimize distance distortion, approximating the lossless qualities of the Oracle-SVD test.
2. **Relation-Aware Encoders:** Swapping the MiniLM encoder for a model explicitly fine-tuned on relational/NLI tasks to raise the semantic manifold ceiling.
3. **Query Disambiguation:** Designing query templates that explicitly prevent relational aliasing (e.g., injecting unique UUIDs or timestamps).
4. **Triton Edge Compilation:** Porting the custom Python autograd adder kernels to Triton/C++ for true edge-deployment speeds.

---

## 📁 Project Structure & Quick Start

```
bioneural-network/
├── mnc_project/
│   ├── mnc/                            # Custom layers & optimizers
│   │   ├── kernels.py                  # Custom L1 forward & L2/HTDR backward primitives
│   │   ├── layers.py                   # MNCLinear Layer definitions
│   │   └── memory.py                   # MESUEngine optimization updates
│   ├── data/journal.txt                # 10-day delayed recall synthetic stream
│   ├── pipeline.py                     # MiniLM embedding encoder pipeline
│   ├── run_comprehensive_validation.py # Full 10-seed, 5-study validation suite
│   ├── run_recall_test.py              # Single-seed recall sanity run
│   └── requirements.txt                # Dependency specifications
├── experiments/                        # Rigorous Scientific Audits (Exp 1 - 5)
├── plans/                              # Architectural blueprints and Context
└── README.md
```

### Installation & Execution

```bash
cd mnc_project
virtualenv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

# Run Sanity Recall Test
python run_recall_test.py

# Run Full 10-Seed Validation Suite
python run_comprehensive_validation.py

# Run Sweeps (Example)
python ../experiments/exp5_unambiguous_scaling/unambiguous_scaling.py --N_max 100
```

---

## 📖 Citation

```bibtex
@software{mnc_framework_2026,
  author = {Sayanthegamer},
  title = {BioNeural Network: Multiplication-Free Continual Learning via Metaplastic Synaptic Uncertainty},
  year = {2026},
  url = {https://github.com/Sayanthegamer/bioneural-network}
}
```
