# Experiment 5: Unambiguous Schema Scaling & True Capacity Validation

## Capacity Breakdown Points ($N^*$)
Defined as the largest $N \in \{100, 200, 400, 800, 1600, 3200\}$ where $\text{Recall@1} \ge 95\%$ and $\text{p5 Margin} \ge 0$.

| Projection Configuration | $N^*$ |
| :--- | :---: |
| Raw (384D) | 100 |
| Random (W=32) | < 100 |
| Random (W=64) | < 100 |
| Random (W=128) | < 100 |
| Oracle-SVD (W=32) | < 100 |
| Oracle-SVD (W=64) | < 100 |
| Oracle-SVD (W=128) | < 100 |


> [!IMPORTANT]
> **Oracle-SVD Data Leakage Disclaimer:** The Oracle-SVD projection matrix is computed based on the same statement corpus being retrieved.
> Therefore, Oracle-SVD results serve as an upper-bound representational ceiling rather than a realistic deployment scenario.

## Detailed Sweep Results

| N | Projection | Width (W) | Recall@1 | Recall@10 | Mean Rank | p5 Margin | Mean Gap Ratio | Mean Top1 Dist | Mean Top2 Dist | Mean Decision Gap | Mean Density |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | Raw | - | 95.0% | 100.0% | 1.1 | 0.03 | 1.16 | 9.94 | 11.42 | 1.47 | 12.13 |
| 100 | Random | 32 | 61.0% ± 3.6% | 96.3% | 2.5 | -0.12 | 1.05 | 0.82 | 0.89 | 0.07 | 0.97 |
| 100 | Oracle-SVD | 32 | 81.0% | 100.0% | 1.3 | -0.07 | 1.18 | 1.58 | 1.84 | 0.26 | 2.13 |
| 100 | Random | 64 | 80.3% ± 1.2% | 99.7% | 1.4 | -0.08 | 1.11 | 1.62 | 1.81 | 0.18 | 1.94 |
| 100 | Oracle-SVD | 64 | 92.0% | 100.0% | 1.1 | -0.02 | 1.29 | 2.59 | 3.26 | 0.67 | 3.65 |
| 100 | Random | 128 | 85.0% ± 1.6% | 100.0% | 1.2 | -0.11 | 1.13 | 3.28 | 3.69 | 0.41 | 3.95 |
| 100 | Oracle-SVD | 128 | 93.0% | 100.0% | 1.1 | -0.04 | 1.23 | 4.53 | 5.48 | 0.95 | 5.86 |