# Experiment Updates and Recalibrations

Following a comprehensive audit and rerun of the `experiments` module, several empirical baseline CSV results inside `experiments/results/` have been updated.

The previous CSVs contained results from runs that were stale compared to the `.py` scripts that generated them (e.g. `u2_ablation.py`, `semantic_interference.py`, `capacity_wall.py`, etc.). The code within these scripts contained recent tweaks (like bug fixes to `engine.zero_grad()`, variance priors, and width scaling properties as documented in the codebase) that were never reflected in the exported data until now.

### Affected Telemetry Data
The following output CSV logs have been updated with fresh simulation data:
*   `capacity_wall.csv`
*   `drift_analysis.csv`
*   `replay_comparison.csv`
*   `semantic_interference.csv`
*   `u2_ablation.csv`
*   `variance_telemetry.csv`

### Notable Changes in Data
1. **Variance Telemetry & Drift Analysis:** Values reflect the corrected PyTorch gradient clearing logic which finally allows un-driven variance tracking via the rest-phase relaxation process.
2. **Replay Comparison:** The data now correctly measures the differential efficiency vs recall rates using a fixed sparse buffer (Size=10, 50, 100) vs a sequential baseline (Size=0). The results perfectly echo the known limitation stated in the README: SGD completely collapses under these regimes.

All experiments are structurally sound and exhibit consistent baseline methodology (fixed dataset seeds vs randomized initialization seeds).
