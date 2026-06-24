# Experiment 3 Summary: Relational Manifold Audit & Oracles

## 🏆 Scientific Verdict

### 1. Cross-Encoder Relational Comparison (Raw Space, Cosine Metric, L2-normalized, N=800)

| Encoder | Recall (Prototype) | Recall (1-NN) | Prototype Advantage | Linear Separability Probe (SVC) | Oracle Centroid (Top-10) | Oracle Exemplar (Top-10) | Median Rank (Proto) | Median Rank (Exemplar) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **all-MiniLM-L6-v2** | 28.1% | 28.1% | +0.0% | 23.1% | 100.0% | 100.0% | 1.0 | 1.0 |
| **all-mpnet-base-v2** | 27.8% | 27.9% | -0.1% | 24.1% | 99.2% | 99.2% | 1.0 | 1.0 |
| **bge-small-en-v1.5** | 28.1% | 28.1% | +0.0% | 21.6% | 100.0% | 99.9% | 1.0 | 1.0 |
| **e5-base-v2** | 28.1% | 28.1% | +0.0% | 27.9% | 99.8% | 99.4% | 1.0 | 1.0 |
| **e5-small-v2** | 27.6% | 28.0% | -0.4% | 27.6% | 99.4% | 99.6% | 1.0 | 1.0 |

### 2. Embedding Normalization Impact (all-MiniLM-L6-v2, Raw Space, L1 Metric, N=800)

| Normalization | Recall (Prototype) | Recall (1-NN) | Prototype Advantage | Median Rank (Proto) | Median Rank (Exemplar) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| raw | 28.1% | 28.1% | +0.0% | 1.0 | 1.0 |
| l2_normalized | 28.1% | 28.1% | +0.0% | 1.0 | 1.0 |

### 3. Projection Distortion vs. Data-Dependent SVD (Diagnostic Upper Bound) (all-MiniLM-L6-v2, N=800, L1 Metric)

| Space | Projection | Recall (Prototype) | Recall Loss | Median Distortion (ε) | 95th Percentile Distortion |
| :--- | :--- | :---: | :---: | :---: | :---: |
| proj_128d | random | 26.0% | 2.1% | 1.0000 | 1.0000 |
| proj_128d | svd | 28.1% | 0.0% | 0.5857 | 0.6267 |
| proj_256d | random | 26.8% | 1.4% | 0.9185 | 0.9339 |
| proj_256d | svd | 28.1% | 0.0% | 0.5280 | 0.5793 |

### 4. Margin CDF Tail Distribution (N=800, Raw Space, Cosine Metric, L2-normalized)

| Encoder | 1st Percentile | 5th Percentile | 10th Percentile | 25th Percentile | 50th Percentile (Median) | 90th Percentile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **all-MiniLM-L6-v2** | -0.0579 | -0.0376 | -0.0283 | -0.0177 | -0.0077 | 0.0080 |
| **all-mpnet-base-v2** | -0.0483 | -0.0369 | -0.0309 | -0.0222 | -0.0085 | 0.0125 |
| **bge-small-en-v1.5** | -0.0367 | -0.0285 | -0.0248 | -0.0162 | -0.0085 | 0.0095 |
| **e5-base-v2** | -0.0176 | -0.0142 | -0.0114 | -0.0076 | -0.0033 | 0.0045 |
| **e5-small-v2** | -0.0270 | -0.0163 | -0.0133 | -0.0084 | -0.0037 | 0.0041 |


## 🔍 Interpretation & Recommendations
Use these findings to update the project roadmap:
1. **Is the manifold structurally weak or is ranking failing?** Compare the Oracle Centroid (Top-10) and Exemplar (Top-10) with the Prototype recall. If Oracle is > 80% while Prototype is < 50%, the information is present and metrically accessible, but distance ranking requires a non-linear or multi-prototype scheme.
2. **Is there an encoder ceiling?** Compare maximum recalls across encoders at N=800. If all encoders plateau at similar levels (< 60%), the relational collision dataset exposes a general geometric packing limit of sentence transformers.
3. **Does L2-Normalization fix metric degradation?** Compare raw vs L2-normalized recall. If E5/MPNet/BGE recall jumps significantly under L2-normalization, cosine similarity on normalized vectors must be enforced throughout the MNC layers.
