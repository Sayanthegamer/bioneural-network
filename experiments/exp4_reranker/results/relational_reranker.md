# Experiment 4: Relational Re-Ranker & Disambiguation Evaluation

**Evaluated at N = [100]**

---

## 1. Overall Results

| N | Coverage@10 | Coverage@20 | Alias Coverage Ratio | Baseline | Token-Max | CE-Max | CE-Mean |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## 2. Per-Template Results (N=100)

| Template | Count | Coverage@10 | Alias Coverage Ratio | Baseline | Token-Max | CE-Max | CE-Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| T0: Access Codes | 34 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| T1: Folders/Projects | 33 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| T2: Meetings | 33 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## 3. Recall vs Alias Cardinality (N=100)

| M | Count | Theoretical Bayes Ceiling | Coverage@10 | Alias Coverage Ratio | Baseline | CE-Max | Δ CE-Max | Δ Token-Max |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 100 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | +0.0pp | +0.0pp |

## 4. Error Decomposition (N=100)

- **Retrieval Error** = 100% - Coverage@10 = 0.0%
- **Selection Error** = Coverage@10 - CE-Max Recall = 0.0%
- **Total Error** = 100% - CE-Max Recall = 0.0%
