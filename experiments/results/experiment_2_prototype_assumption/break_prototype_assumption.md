# Experiment 2 Summary: Breaking the Prototype Assumption

## 🏆 Scientific Verdict

### 1. Retrieval Comparison for N=400 and N=800

| Space | N | Regime | Prototype Recall | 1-NN Recall | Oracle Recall | Delta Margin (p5) | Lcomp Ratio | Silhouette |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **raw_384d** | 400 | Unique ID | 96.5% | 94.5% | 94.5% | 0.0843 | 1.058 | 0.000 |
| **raw_384d** | 400 | Relational | 64.5% | 63.7% | 63.7% | -0.6549 | 0.972 | 0.000 |
| **raw_384d** | 800 | Unique ID | 92.6% | 90.0% | 90.0% | -0.1356 | 1.054 | 0.000 |
| **raw_384d** | 800 | Relational | 46.2% | 46.0% | 46.0% | -0.9111 | 0.971 | 0.000 |
| **proj_128d** | 400 | Unique ID | 64.0% | 57.2% | 57.2% | -0.0000 | 1.049 | 0.000 |
| **proj_128d** | 400 | Relational | 53.5% | 53.2% | 53.2% | -0.0000 | 1.021 | 0.000 |
| **proj_128d** | 800 | Unique ID | 57.5% | 51.0% | 51.0% | -0.0000 | 1.049 | 0.000 |
| **proj_128d** | 800 | Relational | 34.0% | 29.5% | 29.5% | -0.0000 | 1.018 | 0.000 |
| **proj_256d** | 400 | Unique ID | 74.8% | 64.5% | 64.5% | -0.0521 | 1.003 | 0.000 |
| **proj_256d** | 400 | Relational | 62.7% | 61.3% | 61.3% | -0.0934 | 1.002 | 0.000 |
| **proj_256d** | 800 | Unique ID | 66.9% | 57.2% | 57.2% | -0.0681 | 0.998 | 0.000 |
| **proj_256d** | 800 | Relational | 41.1% | 39.1% | 39.1% | -0.1419 | 0.999 | 0.000 |

### 2. Auxiliary Metric Control (Raw 384D Space, N=800 & 1600)

| N Facts | Regime | Metric | Prototype Recall | 1-NN Recall |
| :---: | :--- | :---: | :---: | :---: |
| 800 | Unique ID | L1 | Prototype | 92.62% |
| 800 | Unique ID | L1 | 1-NN | 90.00% |
| 800 | Unique ID | L2 | Prototype | 93.00% |
| 800 | Unique ID | L2 | 1-NN | 90.75% |
| 800 | Unique ID | Cosine | Prototype | 92.88% |
| 800 | Unique ID | Cosine | 1-NN | 90.75% |
| 1600 | Unique ID | L1 | Prototype | 88.06% |
| 1600 | Unique ID | L1 | 1-NN | 84.50% |
| 1600 | Unique ID | L2 | Prototype | 88.62% |
| 1600 | Unique ID | L2 | 1-NN | 85.88% |
| 1600 | Unique ID | Cosine | Prototype | 88.62% |
| 1600 | Unique ID | Cosine | 1-NN | 85.88% |
| 800 | Relational | L1 | Prototype | 46.25% |
| 800 | Relational | L1 | 1-NN | 46.00% |
| 800 | Relational | L2 | Prototype | 46.75% |
| 800 | Relational | L2 | 1-NN | 44.75% |
| 800 | Relational | Cosine | Prototype | 46.75% |
| 800 | Relational | Cosine | 1-NN | 44.75% |
| 1600 | Relational | L1 | Prototype | 35.12% |
| 1600 | Relational | L1 | 1-NN | 32.75% |
| 1600 | Relational | L2 | Prototype | 35.00% |
| 1600 | Relational | L2 | 1-NN | 31.37% |
| 1600 | Relational | Cosine | Prototype | 34.94% |
| 1600 | Relational | Cosine | 1-NN | 31.37% |


## 🔍 Interpretation & Recommendations
Evaluate the pre-hoc decision criteria:
1. **Is there retrieval compression loss?** Check the difference between Oracle/1-NN and Prototype recall. If it is < 10% and Lcomp is close to 1.0, then single-centroid prototype compression is not the main bottleneck.
2. **Is there a projection bottleneck?** Compare Raw 384D recall against Projected 128D/256D at identical fact horizon N. If the gap is > 10%, the random linear projection destroys manifold alignment.
3. **Is the manifold itself saturating?** If Raw 384D recall across all metrics/retrievals remains < 85% at N=800, the encoder's representational boundaries are overlapping.
