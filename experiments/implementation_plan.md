# Implementation Plan: Methodological Rigor & Unambiguous Schema Scaling (Experiment 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Experiment 5 (Unambiguous Schema Scaling & True Capacity Validation) to measure the true representation capacity of the prototypical network up to N=3200 classes using an alias-free schema, and refine Experiment 4's metrics.

**Architecture:** A sweep script that embeds facts via frozen MiniLM, runs prototypical retrieval with random and SVD-initialized projections, computes gap ratios, prototype crowding density, and capacity breakdown point $N^*$, and plots metrics without changing the core `mnc/layers.py` package.

**Tech Stack:** Python, PyTorch, SentenceTransformers, Matplotlib, NumPy.

## Global Constraints
* Do not modify `mnc/layers.py` inside the core package.
* Use `sentence-transformers/all-MiniLM-L6-v2` for local sentence encoding.
* Ensure all telemetry labels strictly map to `Coverage@K` (instead of `Oracle@K`) and `Theoretical Bayes Ceiling` (instead of `Bayes Ceiling`).
* Set random seeds explicitly for reproducibility (seeds 42, 101, 202).

---

## Technical Specifications

### 1. Mathematical Projection Definitions
Rather than modifying or invoking `MNCLinear` architecture layers, the bottleneck projections will be computed using direct tensor operations to keep Experiment 5 mathematically pure:
* **Random Projection**:
  $$Y = X R^T$$
  where $X \in \mathbb{R}^{B \times 384}$ is the input embedding matrix, and $R \in \mathbb{R}^{W \times 384}$ is a random projection matrix whose rows are initialized from a normal distribution and normalized to the unit sphere: $R_i \leftarrow R_i / \|R_i\|_2$.
* **SVD Projection**:
  $$Y = X V_W^T$$
  where $V_W \in \mathbb{R}^{W \times 384}$ is the matrix containing the top $W$ right-singular vectors of the statement embedding matrix.
  > [!IMPORTANT]
  > **SVD Data Leakage Disclaimer:** SVD projection is fitted on the same statement corpus used for retrieval evaluation. Therefore, SVD results should be interpreted as an upper-bound representation benchmark rather than a strictly deployment-realistic configuration. This limitation must be explicitly documented in the final report.

### 2. Rank Definition
For each query $q$:
$$\text{Rank}(q) = \text{position of the correct class prototype after sorting all } N \text{ prototypes by ascending L1 distance.}$$
* A Rank of 1 represents a perfect retrieval (the correct class is the closest).

### 3. Prototype Crowding Telemetry
To measure geometric crowding directly:
$$\text{prototype\_density}(q) = \text{mean distance of query } q \text{ to the 10 nearest prototypes.}$$
We track:
* `mean_prototype_density` (Mean across all queries)
* `p5_prototype_density`
* `p10_prototype_density`

### 4. Capacity Breakpoint Definition ($N^*$)
$$N^* = \text{largest evaluated } N \in \{100, 200, 400, 800, 1600, 3200\} \text{ satisfying: } \text{Recall@1} \ge 95\% \text{ AND } \text{p5 Margin} \ge 0$$
* If $N=100$ does not satisfy the conditions, $N^* < 100$.

---

## Proposed Changes

### Component 1: Experiment 4 Codebase Polish & Telemetry Renames

Modify the existing Experiment 4 script and its Kaggle counterpart to incorporate rename requests and new coverage ratio telemetry.

#### [MODIFY] [relational_reranker.py](file:///c:/Users/Anon/Downloads/Northstar/experiments/exp4_reranker/relational_reranker.py)
* Rename metrics:
  - `oracle_at_10` $\to$ `coverage_at_10`
  - `oracle_at_20` $\to$ `coverage_at_20`
  - `Oracle@10` $\to$ `Coverage@10`
  - `Oracle@20` $\to$ `Coverage@20`
  - `Bayes Ceiling` $\to$ `Theoretical Bayes Ceiling`
* Add `alias_coverage_ratio` telemetry. For each query, `alias_coverage_ratio = (number of retrieved aliases in Top-K) / len(aliases)`.
* Update CSV output columns, screen logging, markdown report, and plots to reflect the rename.

#### [MODIFY] [kaggle_relational_reranker.py](file:///c:/Users/Anon/Downloads/Northstar/experiments/kaggle/kaggle_relational_reranker.py)
* Mirror the identical metric renames and `alias_coverage_ratio` telemetry inside the Kaggle script.

---

### Component 2: Experiment 5 Implementation (Unambiguous Schema Scaling)

Create the new unambiguous scaling sweep script and its Kaggle template.

#### [NEW] [unambiguous_scaling.py](file:///c:/Users/Anon/Downloads/Northstar/experiments/exp1_width_scaling/unambiguous_scaling.py)
A self-contained Python script to execute the Experiment 5 sweep:
1. **Alias-Free Generator**:
   - Pools: `rooms` (100 names), `entrances` (100 names), `colors` (100 names), `projects` (100 names), `coordinators` (100 names), `sessions` (100 names).
   - Generates statements and queries ensuring `max_alias == 1` and `duplicate_queries == 0`.
2. **Sweep Configurations**:
   - Sweeps $N \in \{100, 200, 400, 800, 1600, 3200\}$ and widths $W \in \{32, 64, 128\}$.
   - Evaluates:
     - **Raw 384D**: Uncompressed bi-encoder representation.
     - **Random Projection**: standard projection to widths $W \in \{32, 64, 128\}$ (averaged over seeds `[42, 101, 202]`).
     - **Oracle-SVD Projection**: Oracle-SVD projection to widths $W \in \{32, 64, 128\}$ using the right-singular vectors of the statement corpus (explicitly renamed to denote corpus adaptation).
3. **Telemetry & Metrics**:
   - Rank-based metrics: `Recall@1`, `Recall@5`, `Recall@10`, and `Mean/Median Rank` (using the sorted ascending L1 distance definition).
   - Nearest-Neighbor Gap Telemetry:
     - `same_class_distance` = L1 distance between query representation and its own class prototype.
     - `nearest_other_distance` = minimum L1 distance from query representation to all other prototypes.
     - `margin` = `nearest_other_distance` - `same_class_distance`.
     - `gap_ratio` = `nearest_other_distance / (same_class_distance + 1e-8)`.
     - Log mean, p5, and p10 for both `margin` and `gap_ratio`.
   - Crowding Telemetry:
     - `prototype_density` = mean L1 distance to the 10 nearest prototypes.
   - Ranking Telemetry:
     - `top1_distance`, `top2_distance`, and `decision_gap` (top2_distance - top1_distance) to measure ranking stability. Log mean.
   - Capacity Breakdown Point $N^*$.
4. **Outputs**:
   - CSV results: `experiments/results/experiment_5_unambiguous_scaling/unambiguous_scaling.csv`.
   - PNG plots: `experiments/results/experiment_5_unambiguous_scaling/unambiguous_scaling.png`.
   - Markdown report: `experiments/results/experiment_5_unambiguous_scaling/unambiguous_scaling.md`.

#### [NEW] [kaggle_unambiguous_scaling.py](file:///c:/Users/Anon/Downloads/Northstar/experiments/kaggle/kaggle_unambiguous_scaling.py)
* A self-contained version of `unambiguous_scaling.py` formatted for Kaggle notebooks (utilizing GPU if available).

---

## Tasks

### Task 1: Refactor and Polish Experiment 4 Telemetry
- [ ] **Step 1: Edit `experiments/exp4_reranker/relational_reranker.py`**
  - Add `alias_coverage_ratio` to the query loop:
    ```python
    aliases = query_to_labels[facts[q_idx]["query"]]
    retrieved_aliases = sum(1 for a in aliases if a in candidates)
    alias_coverage_ratio = retrieved_aliases / len(aliases)
    ```
  - Rename all occurrences of `Oracle@K` to `Coverage@K` and `Bayes Ceiling` to `Theoretical Bayes Ceiling`.
  - Update `write_csv()`, `generate_plots()`, and `generate_report()` to handle the renamed variables.
- [ ] **Step 2: Run verification test on relational_reranker.py**
  Run: `mnc_project\venv\Scripts\python.exe experiments/exp4_reranker/relational_reranker.py --N_max 100`
  Expected: Successful run, output files updated in `experiments/results/experiment_4_reranker/` with correct renames and headers.
- [ ] **Step 3: Edit `experiments/kaggle/kaggle_relational_reranker.py`**
  - Update the Kaggle template script with the same telemetry additions and metric renames.
- [ ] **Step 4: Commit Experiment 4 refactoring**
  Run: `git add experiments/exp4_reranker/relational_reranker.py experiments/kaggle/kaggle_relational_reranker.py` and commit with message `refactor: update exp4 reranker metrics and telemetry`.

### Task 2: Implement Experiment 5 (Unambiguous Schema Scaling)
- [ ] **Step 1: Write `experiments/exp1_width_scaling/unambiguous_scaling.py`**
  - Implement the alias-free generator (pools of 100).
  - Implement Random (variance-controlled with seeds `[42, 101, 202]`) and Oracle-SVD projection mathematics.
  - Calculate `Recall@1`, `Recall@5`, `Recall@10`, distance margins, `gap_ratio` (mean, p5, p10), `prototype_density`, and ranking telemetry (`top1_distance`, `top2_distance`, `decision_gap`).
  - Compute the capacity breakdown point $N^*$.
  - Add plotting and reporting logic.
- [ ] **Step 2: Run local micro-validation at N=100**
  Run: `mnc_project\venv\Scripts\python.exe experiments/exp1_width_scaling/unambiguous_scaling.py --N_max 100`
  Expected: Runs successfully, generates CSV, png, and markdown report under `experiments/results/experiment_5_unambiguous_scaling/`.
- [ ] **Step 3: Create Kaggle script `experiments/kaggle/kaggle_unambiguous_scaling.py`**
  - Save a copy of the self-contained script optimized for single-cell Kaggle running.
- [ ] **Step 4: Commit Experiment 5**
  Run: `git add experiments/exp1_width_scaling/unambiguous_scaling.py experiments/kaggle/kaggle_unambiguous_scaling.py` and commit with message `feat: implement exp5 unambiguous scaling sweep and SVD projection`.

---

## Verification Plan

### Automated Tests
1. **Experiment 4 Local Run**:
   `mnc_project\venv\Scripts\python.exe experiments/exp4_reranker/relational_reranker.py --N_max 100`
2. **Experiment 5 Local Run**:
   `mnc_project\venv\Scripts\python.exe experiments/exp1_width_scaling/unambiguous_scaling.py --N_max 100`

### Manual Verification
1. Inspect the generated output CSV and plots to confirm that `Recall@1` is listed explicitly, the gap ratio is tracked, and the capacity breakdown point $N^*$ is calculated and plotted.
2. Verify that `mnc/layers.py` remains unmodified.
