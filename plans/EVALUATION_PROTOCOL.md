# 10-Day Sequential Evaluation Protocol
**CRITICAL RULE:** Standard PyTorch batched training loops and `DataLoaders` are strictly forbidden for this evaluation.

## 1. The Ingestion Constraint
* **Batch Size:** Exactly 1.
* **Storage:** No historical memory buffers, vector databases, or replay loops are permitted. Once a sentence vector completes its backward pass and weight update, the tensor must be permanently deleted from memory.
* **Embedding:** Text must be embedded using `sentence-transformers` (`all-MiniLM-L6-v2`) on the CPU prior to passing to the MNC.

## 2. The Timeline
The validation script must simulate a 10-day sequence using a synthetic personal journal dataset:
* **Days 1-5 (The Consolidated Context):** The network sequentially ingests baseline facts (e.g., spatial locations of objects).
* **Days 6-9 (The Interfering Context):** The network ingests distracting, chaotic events designed to force catastrophic forgetting.
* **Day 10 (The Validation):** No learning occurs. The network is queried exclusively on facts established in Days 1-5.

## 3. The Baseline Control
To prove the MNC works, a standard Decoder-only Transformer ($\approx 5.2M$ parameters) must be subjected to the exact same sequential, unbatched ingestion process. 

## 4. Pass/Fail Metrics
* **Pass:** Day 10 recall accuracy $\ge 85\%$ of the Day 5 post-training baseline accuracy (evaluated immediately after Day 5 training completes, prior to any Day 6–9 interference).
* **Fail:** Day 10 recall accuracy $< 70\%$ of the Day 5 baseline accuracy.