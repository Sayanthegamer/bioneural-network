# Experiment 6: Learned & SVD Projections Audit Report (Full Sweep)

## 1. Key Performance Summary (Bottleneck W=128)

| Mode | N | Identity (384D) | Random (W=128) | SVD (W=128) | Learned (W=128) | Neighbor Recall (Learned) | L1 Distortion (Learned) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Capacity | 100 | 95.0% | 85.0% | 93.0% | 91.0% | 73.5% | 0.0571 |
| Capacity | 200 | 95.0% | 85.8% | 93.5% | 90.5% | 74.3% | 0.0591 |
| Capacity | 400 | 92.5% | 82.1% | 94.5% | 92.5% | 77.2% | 0.0621 |
| Capacity | 800 | 93.0% | 83.8% | 93.2% | 88.0% | 70.0% | 0.0617 |
| Capacity | 1600 | 92.6% | 80.8% | 90.4% | 83.4% | 66.0% | 0.0618 |
| Capacity | 3200 | 92.6% | 79.2% | 92.7% | 79.6% | 59.9% | 0.0643 |
| Generalization | 100 | 98.0% | 94.7% | 98.0% | 94.0% | 77.6% | 0.0463 |
| Generalization | 200 | 95.0% | 89.3% | 93.0% | 90.0% | 73.0% | 0.0441 |
| Generalization | 400 | 90.0% | 79.8% | 90.5% | 89.5% | 71.2% | 0.0476 |
| Generalization | 800 | 93.5% | 85.7% | 94.8% | 86.2% | 65.2% | 0.0459 |
| Generalization | 1600 | 93.4% | 82.3% | 94.1% | 86.0% | 68.3% | 0.0550 |
| Generalization | 3200 | 93.1% | 80.4% | 88.2% | 83.1% | 66.7% | 0.0550 |

## 2. Margin & Telemetry Details (W=128)

| Mode | N | Baseline | Margin (Mean) | Margin (P5) | Decision Gap | 
| :---: | :---: | :---: | :---: | :---: | :---: |
| Capacity | 100 | Identity | 1.45 | 0.03 | 1.47 |
| Capacity | 100 | Random | 0.38 | -0.11 | 0.41 |
| Capacity | 100 | Svd | 0.94 | -0.04 | 0.95 |
| Capacity | 100 | Learned | 1.25 | -0.11 | 1.28 |
| Capacity | 200 | Identity | 1.48 | 0.03 | 1.50 |
| Capacity | 200 | Random | 0.40 | -0.10 | 0.42 |
| Capacity | 200 | Svd | 0.95 | -0.04 | 0.97 |
| Capacity | 200 | Learned | 0.87 | -0.09 | 0.90 |
| Capacity | 400 | Identity | 1.31 | -0.13 | 1.35 |
| Capacity | 400 | Random | 0.34 | -0.16 | 0.38 |
| Capacity | 400 | Svd | 0.78 | -0.05 | 0.80 |
| Capacity | 400 | Learned | 0.75 | -0.10 | 0.78 |
| Capacity | 800 | Identity | 1.28 | -0.10 | 1.31 |
| Capacity | 800 | Random | 0.34 | -0.14 | 0.38 |
| Capacity | 800 | Svd | 0.72 | -0.02 | 0.73 |
| Capacity | 800 | Learned | 0.71 | -0.19 | 0.76 |
| Capacity | 1600 | Identity | 1.22 | -0.08 | 1.25 |
| Capacity | 1600 | Random | 0.32 | -0.19 | 0.36 |
| Capacity | 1600 | Svd | 0.74 | -0.11 | 0.76 |
| Capacity | 1600 | Learned | 0.65 | -0.35 | 0.73 |
| Capacity | 3200 | Identity | 1.16 | -0.07 | 1.19 |
| Capacity | 3200 | Random | 0.29 | -0.19 | 0.34 |
| Capacity | 3200 | Svd | 0.71 | -0.05 | 0.72 |
| Capacity | 3200 | Learned | 0.58 | -0.39 | 0.68 |
| Generalization | 100 | Identity | 1.84 | 0.57 | 1.87 |
| Generalization | 100 | Random | 0.52 | 0.03 | 0.54 |
| Generalization | 100 | Svd | 0.78 | 0.32 | 0.79 |
| Generalization | 100 | Learned | 1.24 | -0.00 | 1.25 |
| Generalization | 200 | Identity | 1.54 | 0.03 | 1.56 |
| Generalization | 200 | Random | 0.43 | -0.09 | 0.45 |
| Generalization | 200 | Svd | 0.76 | -0.04 | 0.78 |
| Generalization | 200 | Learned | 0.92 | -0.13 | 0.95 |
| Generalization | 400 | Identity | 1.38 | -0.18 | 1.43 |
| Generalization | 400 | Random | 0.38 | -0.19 | 0.43 |
| Generalization | 400 | Svd | 0.86 | -0.13 | 0.89 |
| Generalization | 400 | Learned | 0.76 | -0.10 | 0.81 |
| Generalization | 800 | Identity | 1.41 | -0.05 | 1.43 |
| Generalization | 800 | Random | 0.40 | -0.12 | 0.43 |
| Generalization | 800 | Svd | 0.82 | -0.02 | 0.83 |
| Generalization | 800 | Learned | 0.69 | -0.17 | 0.73 |
| Generalization | 1600 | Identity | 1.28 | -0.06 | 1.30 |
| Generalization | 1600 | Random | 0.35 | -0.19 | 0.39 |
| Generalization | 1600 | Svd | 0.66 | -0.00 | 0.67 |
| Generalization | 1600 | Learned | 0.67 | -0.30 | 0.74 |
| Generalization | 3200 | Identity | 1.30 | -0.04 | 1.31 |
| Generalization | 3200 | Random | 0.34 | -0.17 | 0.38 |
| Generalization | 3200 | Svd | 0.77 | -0.12 | 0.80 |
| Generalization | 3200 | Learned | 0.70 | -0.31 | 0.76 |
