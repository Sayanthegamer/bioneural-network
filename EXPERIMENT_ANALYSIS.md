
## MNC Project Codebase Overview
- The `mnc_project` directory contains the actual framework code.
- The models use a custom component `MNCLinear` and a custom training controller `MESUEngine` (Metaplasticity from Synaptic Uncertainty).
- The `pipeline.py` integrates a frozen local CPU SentenceTransformer to generate 384-dimensional embeddings.
- **Autocalibration Method:** The scaling distances used dynamically inside layers are correctly calibrated (`autocalibrate_scale_distances`). They fix the seed to 42 for this specific sub-routine so it remains deterministic.

Overall, the separation of concerns between `mnc_project/` containing the core logic and `experiments/` containing the testing sweeps is clean. No bugs discovered in the engine step tracking.
