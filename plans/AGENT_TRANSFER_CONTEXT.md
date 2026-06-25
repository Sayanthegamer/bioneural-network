# Antigravity Agent Transfer Context (Verbose Memory Dump)

This document serves as the absolute, single-source-of-truth memory vault and transfer protocol for migrating the **Prototypical Memory Relational Capacity** project. Pointing any new Antigravity agent (2.0 or CLI) to this workspace and initiating with this document will bootstrap the agent with 100% of the project's scientific findings, mathematical definitions, codebase layout, and active plans.

---

## 🔬 1. Project Background & Core Architecture

We are researching the representational and storage capacity limits of the **Multiplication-Free Metaplastic Neuro-Channel (MNC)** architecture. 

### A. Core Layers (`mnc_project/mnc/layers.py`)
1. **`MNCLinear` (Multiplication-Free Linear Layer):**
   * Operates as an **Associative Distance Bank**. 
   * Instead of standard matrix multiplication, it computes the negative L1 distance between the input vector $x$ and a matrix of spatial templates $W$:
     $$\text{output} = \text{mnc\_adder}(x, W) + \text{bias}$$
   * Avoids floating-point multiplications, designed for custom hardware compatibility.
2. **`MNCPrototypicalNetwork` (Readout Head):**
   * Wraps a frozen representation backbone.
   * Dynamically accumulates class representations online. When a statement is ingested, it calculates its bottleneck embedding $z$ and updates a running average prototype vector for that class:
     $$\mu_c \leftarrow \frac{n \cdot \mu_c + z}{n + 1}$$
   * During inference/retrieval, queries are embedded and compared against stored prototypes using negative L1 distance.

### B. Ingestion Pipeline (`mnc_project/pipeline.py`)
* The `JournalPipeline` uses a frozen CPU-bound `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` to generate 384-dimensional sentence embeddings.
* Enforces sequential ingestion constraints (Batch Size = 1).

---

## 📈 2. Historical Progress & Findings

We have conducted four progressive experiments to trace where and why prototypical memory retrieval fails as facts scale.

### 🧪 Experiment 1: Prototypical Bottleneck Width Scaling Sweep
* **Objective:** Map memory recall performance as the number of stored facts $N$ grows and bottleneck width $W$ varies ($W \in \{32, 64, 128, 256, 512\}$).
* **Setup:** Swept facts from $N=5$ up to $N=1600$.
* **Key Finding:** Discovered a power-law recall decay:
  $$\text{Recall}(N) = A / N^\alpha$$
  with a persistent decay exponent $\alpha \approx 1.20$ regardless of bottleneck dimension. This indicated that prototypical memory was experiencing interference-dominated forgetting or representation limits.
* **Paths:** Code is at [prototypical_width_scaling.py](file:///c:/Users/Anon/Downloads/Northstar/experiments/exp1_width_scaling/prototypical_width_scaling.py), telemetry is in `experiments/results/experiment_1_width_scaling/`.

### 🧪 Experiment 2: Prototype Centroid Assumption Audit
* **Objective:** Audit whether averaging sentence embeddings into a single prototype centroid introduces geometric distortion compared to a raw instance-based 1-NN search.
* **Key Finding:** Centroid prototype classification performed nearly identically to instance-based 1-NN lookup. This mathematically validated the prototypical readout head design, proving centroids are effective descriptors.
* **Paths:** Code is at [constant_density_test.py](file:///c:/Users/Anon/Downloads/Northstar/experiments/exp1_width_scaling/constant_density_test.py) and other scripts under `experiments/exp2_prototype_assumption/`.

### 🧪 Experiment 3: Relational Manifold Audit
* **Objective:** Audit prototypical memory under three structured relational schemas representing different types of real-world facts:
  * **Template 0 (Access Codes):** "The access code for server room {room} {entrance} is {num}."
  * **Template 1 (Folders/Projects):** "The {color} folder for Project {project} is in {loc}."
  * **Template 2 (Meetings):** "{coord}'s {session} meeting is {day} at {hour}."
* **The "28% Recall Wall":** In Experiment 3, Template 1 recall crashed to $\sim 28\%$ even at small $N$, failing to scale.
* **Key Finding:** We discovered **Relational Aliasing**. The dataset generator constrained color index assignment to $color\_idx \% 3 == 1$ (leaving only 5 possible colors). Combined with 15 projects, this meant only $5 \times 15 = 75$ unique queries could ever be generated. At $N=800$, this forced an average alias set size of $M \approx 3.56$ distinct facts sharing the *exact same query text*. Because the query text itself was identical, no retrieval method could resolve the target fact, capping recall at the **Theoretical Bayes Ceiling** of $1/M \approx 28.1\%$.
* **Paths:** Code is under `experiments/exp3_relational_audit/`.

### 🧪 Experiment 4: Relational Reranker & Disambiguation Evaluation
* **Objective:** Evaluate if a second-stage reranker (token-matching and Cross-Encoder) can resolve the relational aliasing. Rerank the Top-K candidates retrieved by the prototype bi-encoder.
* **Key Finding:** The Cross-Encoder (and token matching) contributed exactly **0.0% gain** over the baseline. The rerankers could not help because the query itself was information-limited (lacked the distinguishing detail).
* **The Falsification Result:** However, **Coverage@10 was 100%** (the correct class was always in the top 10 retrieved candidates). This proved the prototypical memory successfully stored and retrieved the candidates. The "capacity collapse" was a query ambiguity ceiling, not a memory storage collapse.
* **Paths:** Code is at [relational_reranker.py](file:///c:/Users/Anon/Downloads/Northstar/experiments/exp4_reranker/relational_reranker.py).

---

## ⚙️ 3. Refined Telemetry & Mathematical Specifications

Before proceeding with the final Experiment 5, we must polish Experiment 4's nomenclature and incorporate new telemetry to measure crowding.

### A. Metric Renames (To avoid misleading terminology):
* **`Oracle@K` $\to$ `Coverage@K`:** Represents candidate retrieval success (whether the target class is present in the Top-$K$).
* **`Bayes Ceiling` $\to$ `Theoretical Bayes Ceiling`:** Represents the limit imposed by query ambiguity ($1/M$).
* **`alias_coverage_ratio` (New):** For each query, track the percentage of its alias set that successfully was retrieved in the Top-$K$ candidate list.

### B. Mathematical Projections (Experiment 5):
SVD and Random projections must be implemented using pure matrix operations inside the script to keep the core `mnc/layers.py` code clean:
1. **Random Projection:**
   $$Y = X R^T$$
   where $X \in \mathbb{R}^{B \times 384}$ is the input embedding matrix, and $R \in \mathbb{R}^{W \times 384}$ is a random projection matrix whose rows are normalized to the unit sphere ($R_i \leftarrow R_i / \|R_i\|_2$).
2. **SVD Projection:**
   $$Y = X V_W^T$$
   where $V_W \in \mathbb{R}^{W \times 384}$ contains the top $W$ right-singular vectors of the statement embedding matrix.
   * *Data Leakage Disclaimer:* SVD projection is fitted on the same statement corpus used for retrieval evaluation. This serves as an upper-bound representation benchmark and must be documented as such in final reports.

### C. Rank Definition:
For each query $q$:
$$\text{Rank}(q) = \text{position of the correct class prototype after sorting all } N \text{ prototypes by ascending L1 distance.}$$

### D. Prototype Crowding Telemetry (`prototype_density`):
To measure physical crowding in the projected manifold:
$$\text{prototype\_density}(q) = \text{mean L1 distance to the 10 nearest prototypes.}$$
We log: `mean_prototype_density`, `p5_prototype_density`, and `p10_prototype_density`.

### E. Capacity Breakpoint Metric ($N^*$):
$$N^* = \text{largest evaluated } N \in \{100, 200, 400, 800, 1600, 3200\} \text{ satisfying: } \text{Recall@1} \ge 95\% \text{ AND } \text{p5 Margin} \ge 0.$$
If $N=100$ does not satisfy the conditions, $N^* < 100$.

---

## 📂 4. Current Repository Layout

The codebase has been refactored and organized into structured folders:
* `experiments/exp1_width_scaling/` - Experiment 1 code.
* `experiments/exp2_prototype_assumption/` - Experiment 2 code.
* `experiments/exp3_relational_audit/` - Experiment 3 code.
* `experiments/exp4_reranker/` - Experiment 4 code (contains `relational_reranker.py`).
* `experiments/diagnostics/` - Preflight diagnostics (contains `diagnostic_prototype_preflight.py`).
* `experiments/kaggle/` - Offline scripts for Kaggle sweeps.
* `experiments/results/` - Centralized folder for CSVs, PNGs, and markdown reports.
* `mnc_project/` - The core PyTorch code repository containing model classes and Adders.

---

## 📋 5. Detailed Step-by-Step Task List

### Task 1: Refactor and Polish Experiment 4 Telemetry
* **Step 1.1: Edit `experiments/exp4_reranker/relational_reranker.py`**
  * Update `evaluate_rerankers()` to calculate `alias_coverage_ratio`:
    ```python
    aliases = query_to_labels[facts[q_idx]["query"]]
    retrieved_aliases = sum(1 for a in aliases if a in candidates)
    alias_coverage_ratio = retrieved_aliases / len(aliases)
    ```
  * Rename metrics inside the results dict and logging:
    * `oracle_at_10` $\to$ `coverage_at_10`
    * `oracle_at_20` $\to$ `coverage_at_20`
    * `Oracle@10` $\to$ `Coverage@10`
    * `Oracle@20` $\to$ `Coverage@20`
    * `Bayes Ceiling` $\to$ `Theoretical Bayes Ceiling`
  * Add `alias_coverage_ratio` to the CSV logging, standard output logs, markdown report generator (`generate_report`), and plotting code (`generate_plots`).
* **Step 1.2: Run verification test on relational_reranker.py**
  * Run locally:
    ```powershell
    # Windows PowerShell (using virtual environment)
    mnc_project\venv\Scripts\python.exe experiments/exp4_reranker/relational_reranker.py --N_max 100
    ```
  * Confirm that files under `experiments/results/experiment_4_reranker/` contain the updated metrics, and stdout outputs the correct labels.
* **Step 1.3: Update Kaggle template**
  * Copy changes into `experiments/kaggle/kaggle_relational_reranker.py`.
* **Step 1.4: Stage and Commit**
  * Run `git add experiments/exp4_reranker/relational_reranker.py experiments/kaggle/kaggle_relational_reranker.py` and commit with message `refactor: update exp4 reranker metrics and telemetry`.

### Task 2: Implement Experiment 5 (Unambiguous Schema Scaling)
* **Step 2.1: Write `experiments/exp5_unambiguous_scaling/unambiguous_scaling.py`**
  * **Fact Generator:** Create an alias-free fact generator utilizing pool sizes of 100 rooms, 100 entrances, 100 colors, etc. yielding 10,000 unique combinations. Assert `max_alias == 1` and `duplicate_queries == 0` during initialization.
  * **Projections:** Implement Random Projection ($Y = X R^T$) and SVD Projection ($Y = X V_W^T$) using pure matrix operations. SVD is computed using `torch.svd` or `np.linalg.svd` on the statement embedding matrix.
  * **Evaluation Metric Sweep:** Sweep $N \in \{100, 200, 400, 800, 1600, 3200\}$ and widths $W \in \{32, 64, 128\}$. Log:
    - `Recall@1`, `Recall@5`, `Recall@10`
    - Mean/Median Rank (ascending distance position)
    - Margin statistics (Mean, p5, p10)
    - Gap ratio statistics (Mean, p5, p10)
    - Prototype density (Mean, p5, p10 distance to 10 nearest prototypes)
    - Calculate and display capacity breakpoint $N^*$ for each width/projection type.
  * **Outputs:** Generate CSV logging under `experiments/results/experiment_5_unambiguous_scaling/unambiguous_scaling.csv`, a matplotlib PNG plot under `experiments/results/experiment_5_unambiguous_scaling/unambiguous_scaling.png`, and a markdown summary with SVD leakage disclosures.
* **Step 2.2: Verify locally at N=100**
  * Run:
    ```powershell
    mnc_project\venv\Scripts\python.exe experiments/exp5_unambiguous_scaling/unambiguous_scaling.py --N_max 100
    ```
* **Step 2.3: Generate Kaggle single-cell script**
  * Save a self-contained notebook copy to `experiments/exp5_unambiguous_scaling/kaggle_unambiguous_scaling.py`.
* **Step 2.4: Stage and Commit**
  * Run `git add` and commit with message `feat: implement exp5 unambiguous scaling sweep and SVD projection`.

---

## 🚀 6. Bootstrapping Prompt for Importing Agent

*Copy and paste this prompt when initiating the new Antigravity session:*

```text
Please read the transfer context file located at plans/AGENT_TRANSFER_CONTEXT.md in the workspace to load the context for our current project. Confirm you have understood the scientific background, the reorganized repository structure, the mathematical specs for Experiment 5, and the tasks. Once ready, begin executing Task 1 (Polish Experiment 4 Telemetry).
```
