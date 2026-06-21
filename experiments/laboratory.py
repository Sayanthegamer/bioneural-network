import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. THE PHYSICS LABORATORY API
# ==========================================
class BaseMemoryModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def ingest(self, vector, label, concept_id):
        raise NotImplementedError

    def query(self, vector, true_label, true_concept_id):
        """
        Must return a dict:
        {
            "pred": predicted_label,
            "pred_concept_id": predicted_concept_id,
            "margin_global": float,
            "margin_intra": float
        }
        NOTE: On failure, margins naturally become negative, representing 
        the exact 'confidence gap' by which the true fact was beaten.
        """
        raise NotImplementedError

# ==========================================
# 2. INFINITE CAPACITY CONTROLS
# ==========================================
class OracleRAM(BaseMemoryModel):
    def __init__(self, input_dim=384):
        super().__init__(input_dim)
        self.memory = {}

    def ingest(self, vector, label, concept_id):
        self.memory[label] = concept_id

    def query(self, vector, true_label, true_concept_id):
        if true_label in self.memory:
            return {"pred": true_label, "pred_concept_id": self.memory[true_label], "margin_global": float('nan'), "margin_intra": float('nan')}
        return {"pred": -1, "pred_concept_id": -1, "margin_global": float('nan'), "margin_intra": float('nan')}

class UnboundedKNN(BaseMemoryModel):
    def __init__(self, input_dim=384):
        super().__init__(input_dim)
        self.keys = []
        self.values = []
        self.concepts = []
        self.label_to_idx = {}
        
        # Caching: Removes PyTorch allocation overhead. 
        # Note: Query evaluation is still computationally O(N^2) overall.
        self._cache_valid = False
        self._key_matrix = None
        self._values_tensor = None
        self._concepts_tensor = None

    def ingest(self, vector, label, concept_id):
        self.label_to_idx[label] = len(self.values)
        self.keys.append(vector.detach().clone())
        self.values.append(label)
        self.concepts.append(concept_id)
        self._cache_valid = False 

    def _update_cache(self):
        if not self._cache_valid and self.keys:
            self._key_matrix = torch.cat(self.keys, dim=0)
            self._values_tensor = torch.tensor(self.values)
            self._concepts_tensor = torch.tensor(self.concepts)
            self._cache_valid = True

    def query(self, vector, true_label, true_concept_id):
        if not self.keys: 
            return {"pred": -1, "pred_concept_id": -1, "margin_global": 0.0, "margin_intra": 0.0}
        
        self._update_cache()
        
        similarities = torch.matmul(self._key_matrix, vector.T).squeeze()
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)
            
        best_idx = torch.argmax(similarities).item()
        pred_label = self._values_tensor[best_idx].item()
        pred_concept_id = self._concepts_tensor[best_idx].item()
        
        # O(1) indexing using label_to_idx map
        correct_idx = self.label_to_idx[true_label]
        sim_correct = similarities[correct_idx].item()
        
        sims_incorrect = similarities.clone()
        sims_incorrect[correct_idx] = float('-inf')
        
        sim_incorrect_global = torch.max(sims_incorrect).item()
        margin_global = sim_correct - sim_incorrect_global
        
        intra_mask = (self._concepts_tensor == true_concept_id)
        intra_mask[correct_idx] = False
        sim_incorrect_intra = torch.max(sims_incorrect[intra_mask]).item() if intra_mask.any() else 0.0
        margin_intra = sim_correct - sim_incorrect_intra
        
        return {
            "pred": pred_label,
            "pred_concept_id": pred_concept_id,
            "margin_global": margin_global, 
            "margin_intra": margin_intra
        }

class SinglePassLinearClassifier(BaseMemoryModel):
    """The Single-Pass Parametric Baseline"""
    def __init__(self, input_dim=384, max_classes=2000):
        super().__init__(input_dim)
        self.linear = nn.Linear(input_dim, max_classes)
        self.optimizer = optim.SGD(self.linear.parameters(), lr=0.1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.label_to_concept = {}

    def ingest(self, vector, label, concept_id):
        self.label_to_concept[label] = concept_id
        self.linear.train()
        self.optimizer.zero_grad()
        logits = self.linear(vector)
        loss = self.loss_fn(logits, torch.tensor([label]))
        loss.backward()
        self.optimizer.step()

    def query(self, vector, true_label, true_concept_id):
        self.linear.eval()
        with torch.no_grad():
            logits = self.linear(vector).squeeze()
            
        seen_labels = list(self.label_to_concept.keys())
        if not seen_labels:
            return {"pred": -1, "pred_concept_id": -1, "margin_global": 0.0, "margin_intra": 0.0}

        valid_logits = logits[seen_labels]
        best_idx = torch.argmax(valid_logits).item()
        pred_label = seen_labels[best_idx]
        pred_concept_id = self.label_to_concept[pred_label]

        sim_correct = logits[true_label].item()
        logits_incorrect = logits.clone()
        logits_incorrect[true_label] = float('-inf')
        
        sim_incorrect_global = torch.max(logits_incorrect[seen_labels]).item()
        margin_global = sim_correct - sim_incorrect_global
        
        intra_labels = [l for l, c in self.label_to_concept.items() if c == true_concept_id and l != true_label]
        sim_incorrect_intra = torch.max(logits_incorrect[intra_labels]).item() if intra_labels else 0.0
        margin_intra = sim_correct - sim_incorrect_intra

        return {
            "pred": pred_label,
            "pred_concept_id": pred_concept_id,
            "margin_global": margin_global,
            "margin_intra": margin_intra
        }

class ReplayLinearClassifier(BaseMemoryModel):
    """The Replay-Enhanced Parametric Baseline (Continual Learning Baseline)"""
    def __init__(self, input_dim=384, max_classes=2000):
        super().__init__(input_dim)
        self.linear = nn.Linear(input_dim, max_classes)
        self.optimizer = optim.SGD(self.linear.parameters(), lr=0.1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.label_to_concept = {}
        # Pre-allocate buffer tensors for high performance CPU operations
        self.replay_x = torch.zeros((max_classes, input_dim))
        self.replay_y = torch.zeros(max_classes, dtype=torch.long)
        self.replay_count = 0

    def ingest(self, vector, label, concept_id):
        self.label_to_concept[label] = concept_id
        
        # Store in pre-allocated replay buffer
        if self.replay_count < self.replay_x.size(0):
            self.replay_x[self.replay_count] = vector.detach().squeeze()
            self.replay_y[self.replay_count] = label
            self.replay_count += 1
        
        self.linear.train()
        self.optimizer.zero_grad()
        
        # Loss on current item
        logits = self.linear(vector)
        loss = self.loss_fn(logits, torch.tensor([label]))
        
        # Replay batch
        if self.replay_count > 1:
            batch_size = min(32, self.replay_count)
            # Fast batch index generation using torch.randint
            batch_idx = torch.randint(0, self.replay_count, (batch_size,))
            
            # Slice pre-allocated tensors
            batch_x = self.replay_x[batch_idx]
            batch_y = self.replay_y[batch_idx]
            
            # Forward pass on batch
            batch_logits = self.linear(batch_x)
            loss_replay = self.loss_fn(batch_logits, batch_y)
            loss = loss + loss_replay
            
        loss.backward()
        self.optimizer.step()

    def query(self, vector, true_label, true_concept_id):
        self.linear.eval()
        with torch.no_grad():
            logits = self.linear(vector).squeeze()
            
        seen_labels = list(self.label_to_concept.keys())
        if not seen_labels:
            return {"pred": -1, "pred_concept_id": -1, "margin_global": 0.0, "margin_intra": 0.0}

        valid_logits = logits[seen_labels]
        best_idx = torch.argmax(valid_logits).item()
        pred_label = seen_labels[best_idx]
        pred_concept_id = self.label_to_concept[pred_label]

        sim_correct = logits[true_label].item()
        logits_incorrect = logits.clone()
        logits_incorrect[true_label] = float('-inf')
        
        sim_incorrect_global = torch.max(logits_incorrect[seen_labels]).item()
        margin_global = sim_correct - sim_incorrect_global
        
        intra_labels = [l for l, c in self.label_to_concept.items() if c == true_concept_id and l != true_label]
        sim_incorrect_intra = torch.max(logits_incorrect[intra_labels]).item() if intra_labels else 0.0
        margin_intra = sim_correct - sim_incorrect_intra

        return {
            "pred": pred_label,
            "pred_concept_id": pred_concept_id,
            "margin_global": margin_global,
            "margin_intra": margin_intra
        }

# ==========================================
# 3. EXACT IDENTITY RETRIEVAL BENCHMARK
# ==========================================
class MemoryBenchmark:
    def __init__(self, input_dim=384):
        self.input_dim = input_dim

    def generate_stream(self, n_facts, facts_per_concept, variance, noise_level):
        num_concepts = max(1, math.ceil(n_facts / facts_per_concept))
        concepts = torch.randn(num_concepts, self.input_dim)
        concepts = torch.nn.functional.normalize(concepts, p=2, dim=1)
        
        base_vectors = []
        labels = []
        concept_ids = []
        
        for i in range(n_facts):
            concept_idx = min(i // facts_per_concept, num_concepts - 1)
            concept_center = concepts[concept_idx]
            
            fact = concept_center + (torch.randn(self.input_dim) * variance)
            base_vectors.append(fact)
            labels.append(i)
            concept_ids.append(concept_idx)
            
        base_vectors = torch.stack(base_vectors)
        base_vectors = torch.nn.functional.normalize(base_vectors, p=2, dim=1)
        labels = torch.tensor(labels)
        concept_ids = torch.tensor(concept_ids)
        
        query_vectors = base_vectors + (torch.randn_like(base_vectors) * noise_level)
        query_vectors = torch.nn.functional.normalize(query_vectors, p=2, dim=1)
        
        return base_vectors, labels, concept_ids, query_vectors

# ==========================================
# 4. EXPERIMENTAL ENGINE
# ==========================================
def run_geometric_limits_lab():
    N = 1000
    variance = 0.05 # Brutal density squeeze
    noise_levels = [0.01, 0.03, 0.05]
    facts_per_concept_list = [10, 50, 200]
    num_seeds = 10
    
    models_to_test = [
        (OracleRAM, "Oracle"),
        (UnboundedKNN, "kNN"),
        (SinglePassLinearClassifier, "Linear-1Pass"),
        (ReplayLinearClassifier, "Linear-Replay")
    ]
    
    benchmark = MemoryBenchmark()

    print(f"\n{'='*130}")
    print(f"EXACT IDENTITY RETRIEVAL LAB | N={N} | Concept Variance={variance} | Seeds={num_seeds}")
    print(f"Testing nearest-neighbor robustness to perturbations (Not deep semantic retrieval)")
    print(f"{'='*130}")

    for noise in noise_levels:
        print(f"\n\n[ QUERY NOISE PERTURBATION: {noise:.2f} ]")
        
        for model_class, name in models_to_test:
            print(f"\n--- Model: {name} ---")
            print(f"{'Facts/Concept':<15} | {'Recall':<8} | {'Same-C / Diff-C':<20} | {'Geo Margin (50/5/1%)':<23} | {'Succ Margin (50/5/1%)'}")
            print("-" * 105)
            
            for fpc in facts_per_concept_list:
                all_recalls = []
                seed_same_c_pcts = []
                seed_diff_c_pcts = []
                total_error_count = 0
                
                all_global_margins = []
                all_intra_margins = []
                correct_global_margins = []
                correct_intra_margins = []
                
                for seed in range(num_seeds):
                    import sys
                    sys.stderr.write(f"\r  Running {name} (fpc={fpc}) | Seed {seed+1}/{num_seeds}...")
                    sys.stderr.flush()
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    
                    model = model_class(input_dim=384)
                    train_vecs, labels, concept_ids, query_vecs = benchmark.generate_stream(
                        N, facts_per_concept=fpc, variance=variance, noise_level=noise
                    )
                    
                    for i in range(N):
                        model.ingest(train_vecs[i:i+1], labels[i].item(), concept_ids[i].item())
                        
                    correct = 0
                    seed_errors = 0
                    seed_same_c_errors = 0
                    seed_diff_c_errors = 0
                    
                    for i in range(N):
                        true_lbl = labels[i].item()
                        true_cid = concept_ids[i].item()
                        
                        res = model.query(query_vecs[i:i+1], true_label=true_lbl, true_concept_id=true_cid)
                        
                        if not np.isnan(res["margin_global"]):
                            all_global_margins.append(res["margin_global"])
                            all_intra_margins.append(res["margin_intra"])
                            
                        if res["pred"] == true_lbl:
                            correct += 1
                            if not np.isnan(res["margin_global"]):
                                correct_global_margins.append(res["margin_global"])
                                correct_intra_margins.append(res["margin_intra"])
                        else:
                            seed_errors += 1
                            if res["pred_concept_id"] == true_cid:
                                seed_same_c_errors += 1
                            else:
                                seed_diff_c_errors += 1
                                
                    all_recalls.append(correct / N)
                    total_error_count += seed_errors
                    
                    if seed_errors > 0:
                        seed_same_c_pcts.append((seed_same_c_errors / seed_errors) * 100)
                        seed_diff_c_pcts.append((seed_diff_c_errors / seed_errors) * 100)
                    
                import sys
                sys.stderr.write("\r" + " " * 80 + "\r")
                sys.stderr.flush()
                mean_r = np.mean(all_recalls)
                
                if seed_same_c_pcts:
                    mean_same = np.mean(seed_same_c_pcts)
                    mean_diff = np.mean(seed_diff_c_pcts)
                    same_c_str = f"{mean_same:.1f}% / {mean_diff:.1f}%"
                else:
                    same_c_str = f"N/A (n={total_error_count})"
                
                if not all_global_margins:
                    g_marg_str = "N/A"
                else:
                    g_p50, g_p05, g_p01 = np.percentile(all_global_margins, [50, 5, 1])
                    g_marg_str = f"{g_p50:>5.2f} / {g_p05:>5.2f} / {g_p01:>5.2f}"
                    
                if not correct_global_margins:
                    s_marg_str = "N/A"
                else:
                    s_p50, s_p05, s_p01 = np.percentile(correct_global_margins, [50, 5, 1])
                    s_marg_str = f"{s_p50:>5.2f} / {s_p05:>5.2f} / {s_p01:>5.2f}"
                
                print(f"{fpc:<15d} | {mean_r*100:>5.1f}%   | {same_c_str:<20} | {g_marg_str:<23} | {s_marg_str}")

if __name__ == "__main__":
    run_geometric_limits_lab()