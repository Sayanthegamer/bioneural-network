# Experiment 4: Relational Re-Ranker & Disambiguation Evaluation

**Evaluated at N = [100, 200, 400, 800]**

---

## 1. Overall Results

| N | Oracle@10 | Oracle@20 | Baseline | Token-Max | CE-Max | CE-Mean |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 200 | 100.0% | 100.0% | 91.5% | 91.5% | 91.5% | 91.5% |
| 400 | 100.0% | 100.0% | 56.2% | 56.2% | 56.2% | 56.2% |
| 800 | 100.0% | 100.0% | 28.1% | 28.1% | 28.1% | 28.1% |

## 2. Per-Template Results (N=800)

| Template | Count | Oracle@10 | Baseline | Token-Max | CE-Max | CE-Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| T0: Access Codes | 267 | 100.0% | 18.7% | 18.7% | 18.7% | 18.7% |
| T1: Folders/Projects | 267 | 100.0% | 28.1% | 28.1% | 28.1% | 28.1% |
| T2: Meetings | 266 | 100.0% | 37.6% | 37.6% | 37.6% | 37.6% |

## 3. Recall vs Alias Cardinality (N=800)

| M | Count | Bayes Ceiling | Oracle@10 | Baseline | CE-Max | Δ CE-Max | Δ Token-Max |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 68 | 50.0% | 100.0% | 50.0% | 50.0% | +0.0pp | +0.0pp |
| 3 | 297 | 33.3% | 100.0% | 33.3% | 33.3% | +0.0pp | +0.0pp |
| 4 | 168 | 25.0% | 100.0% | 25.0% | 25.0% | +0.0pp | +0.0pp |
| 5 | 165 | 20.0% | 100.0% | 20.0% | 20.0% | +0.0pp | +0.0pp |
| 6 | 102 | 16.7% | 100.0% | 16.7% | 16.7% | +0.0pp | +0.0pp |

## 4. Error Decomposition (N=800)

- **Retrieval Error** = 100% - Oracle@10 = 0.0%
- **Selection Error** = Oracle@10 - CE-Max Recall = 71.9%
- **Total Error** = 100% - CE-Max Recall = 71.9%
