# MESU Experiment Telemetry Schema

To ensure all stress-testing experiments output compatible, machine-parseable data for comparison and plotting, every run must log its results to `experiments/results/` as a row in a standardized CSV document matching this schema:

| Column | Type | Description |
| :--- | :---: | :--- |
| `seed` | `int` | Random seed used for the evaluation run. |
| `experiment` | `string` | Name of the experiment (e.g. `u2_ablation`, `capacity_wall`). |
| `config` | `string` | Specific configuration key (e.g. `u2_enabled=True`, `num_facts=50`). |
| `alpha_decay` | `float` | Optional. The alpha decay parameter value (absent in some studies like `u2_ablation`). |
| `u2_enabled` | `boolean` | Optional. True if the slow-timescale u2 cascade is enabled. |
| `replay_buffer_size` | `int` | Optional. Size of the replay buffer used. |
| `num_facts` | `int` | Optional. Number of sequential facts used in training. |
| `recall` | `float` | Final recall accuracy on paraphrased queries (0.0 to 1.0). |
| `forgetting` | `float` | Forgetting rate $F = A_{\text{initial}} - A_{\text{final}}$ (0.0 to 1.0). |
| `drift` | `float` | Mean parameter weight drift $\|W_{\text{final}} - W_{\text{initial}}\|_2$ for layer templates. (Replaced by specific drift metrics in `drift_analysis`). |
| `variance_mean` | `float` | Average parameter variance ($\sigma^2$) across all parameters at the end of the run. |
| `variance_min` | `float` | Minimum parameter variance ($\sigma^2$) across the network at the end of the run. |
| `variance_max` | `float` | Maximum parameter variance ($\sigma^2$) across the network at the end of the run. |
| `runtime_sec` | `float` | Time taken to run the evaluation in seconds. |

### Experiment-Specific Columns
Some experiments emit additional columns relevant to their specific analysis:
- **`efficiency`**: Tracked in `replay_comparison`, represents relative compute efficiency vs full replay.
- **`width`**: Tracked in `width_scaling` and `u2_ablation_scaling`, represents the layer width.
- **`drift_active_from_w0` / `drift_dormant_from_w0` / `drift_layer0`**: Specific compartmentalized drift metrics used in `drift_analysis.csv` instead of the general `drift` column.
