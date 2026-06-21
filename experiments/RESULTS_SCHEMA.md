# MESU Experiment Telemetry Schema

To ensure all stress-testing experiments output compatible, machine-parseable data for comparison and plotting, every run must log its results to `experiments/results/` as a row in a standardized CSV document matching this schema:

| Column | Type | Description |
| :--- | :---: | :--- |
| `seed` | `int` | Random seed used for the evaluation run. |
| `experiment` | `string` | Name of the experiment (e.g. `u2_ablation`, `capacity_wall`). |
| `config` | `string` | Specific configuration key (e.g. `u2_enabled=True`, `num_facts=50`). |
| `alpha_decay` | `float` | The alpha decay parameter value (or empty if not applicable). |
| `u2_enabled` | `boolean` | True if the slow-timescale u2 cascade is enabled (or empty if not applicable). |
| `replay_buffer_size` | `int` | Size of the replay buffer used (or empty if not applicable). |
| `num_facts` | `int` | Number of sequential facts used in training (or empty if not applicable). |
| `recall` | `float` | Final recall accuracy on paraphrased queries (0.0 to 1.0). |
| `forgetting` | `float` | Forgetting rate $F = A_{\text{initial}} - A_{\text{final}}$ (0.0 to 1.0). |
| `drift` | `float` | Mean parameter weight drift $\|W_{\text{final}} - W_{\text{initial}}\|_2$ for layer templates. |
| `variance_mean` | `float` | Average parameter variance ($\sigma^2$) across all parameters at the end of the run. |
| `variance_min` | `float` | Minimum parameter variance ($\sigma^2$) across the network at the end of the run. |
| `variance_max` | `float` | Maximum parameter variance ($\sigma^2$) across the network at the end of the run. |
| `runtime_sec` | `float` | Time taken to run the evaluation in seconds. |
