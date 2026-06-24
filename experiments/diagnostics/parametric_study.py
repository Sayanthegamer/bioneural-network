import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
import argparse
import os
import sys
import csv
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==========================================
# DUAL LOG WRITER
# ==========================================
class DualWriter:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.log = open(file_path, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ==========================================
# 1. BASE MEMORY MODEL & PRIMITIVES
# ==========================================
class BaseMemoryModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def ingest(self, vector, label, concept_id):
        raise NotImplementedError

    def query(self, vector, true_label, true_concept_id):
        raise NotImplementedError

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
    def __init__(self, input_dim=384, max_classes=1000):
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
    def __init__(self, input_dim=384, max_classes=1000, buffer_size=200):
        super().__init__(input_dim)
        self.linear = nn.Linear(input_dim, max_classes)
        self.optimizer = optim.SGD(self.linear.parameters(), lr=0.1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.label_to_concept = {}
        self.buffer_size = buffer_size
        
        if self.buffer_size > 0:
            self.replay_x = torch.zeros((buffer_size, input_dim))
            self.replay_y = torch.zeros(buffer_size, dtype=torch.long)
        self.replay_count = 0

    def ingest(self, vector, label, concept_id):
        self.label_to_concept[label] = concept_id
        
        # Buffer insertion
        if self.buffer_size > 0:
            if self.replay_count < self.buffer_size:
                self.replay_x[self.replay_count] = vector.detach().squeeze()
                self.replay_y[self.replay_count] = label
                self.replay_count += 1
            else:
                replace_idx = torch.randint(0, self.buffer_size, (1,)).item()
                self.replay_x[replace_idx] = vector.detach().squeeze()
                self.replay_y[replace_idx] = label
        
        self.linear.train()
        self.optimizer.zero_grad()
        
        logits = self.linear(vector)
        loss = self.loss_fn(logits, torch.tensor([label]))
        
        if self.buffer_size > 0 and self.replay_count > 1:
            batch_size = min(32, self.replay_count)
            batch_idx = torch.randint(0, self.replay_count, (batch_size,))
            batch_x = self.replay_x[batch_idx]
            batch_y = self.replay_y[batch_idx]
            
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

class MLP1HiddenClassifier(BaseMemoryModel):
    def __init__(self, input_dim=384, hidden_dim=128, max_classes=1000):
        super().__init__(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_classes)
        )
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.label_to_concept = {}

    def ingest(self, vector, label, concept_id):
        self.label_to_concept[label] = concept_id
        self.net.train()
        self.optimizer.zero_grad()
        logits = self.net(vector)
        loss = self.loss_fn(logits, torch.tensor([label]))
        loss.backward()
        self.optimizer.step()

    def query(self, vector, true_label, true_concept_id):
        self.net.eval()
        with torch.no_grad():
            logits = self.net(vector).squeeze()
            
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

class ReplayMLP1HiddenClassifier(BaseMemoryModel):
    def __init__(self, input_dim=384, hidden_dim=128, max_classes=1000, buffer_size=200):
        super().__init__(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_classes)
        )
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.label_to_concept = {}
        self.buffer_size = buffer_size
        
        if self.buffer_size > 0:
            self.replay_x = torch.zeros((buffer_size, input_dim))
            self.replay_y = torch.zeros(buffer_size, dtype=torch.long)
        self.replay_count = 0

    def ingest(self, vector, label, concept_id):
        self.label_to_concept[label] = concept_id
        
        if self.buffer_size > 0:
            if self.replay_count < self.buffer_size:
                self.replay_x[self.replay_count] = vector.detach().squeeze()
                self.replay_y[self.replay_count] = label
                self.replay_count += 1
            else:
                replace_idx = torch.randint(0, self.buffer_size, (1,)).item()
                self.replay_x[replace_idx] = vector.detach().squeeze()
                self.replay_y[replace_idx] = label
                
        self.net.train()
        self.optimizer.zero_grad()
        
        logits = self.net(vector)
        loss = self.loss_fn(logits, torch.tensor([label]))
        
        if self.buffer_size > 0 and self.replay_count > 1:
            batch_size = min(32, self.replay_count)
            batch_idx = torch.randint(0, self.replay_count, (batch_size,))
            batch_x = self.replay_x[batch_idx]
            batch_y = self.replay_y[batch_idx]
            
            batch_logits = self.net(batch_x)
            loss_replay = self.loss_fn(batch_logits, batch_y)
            loss = loss + loss_replay
            
        loss.backward()
        self.optimizer.step()
        
    def query(self, vector, true_label, true_concept_id):
        self.net.eval()
        with torch.no_grad():
            logits = self.net(vector).squeeze()
            
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
# 2. OFFLINE CONTROL PRIMITIVES (EARLY STOPPING)
# ==========================================
class OfflineLinearClassifier(BaseMemoryModel):
    def __init__(self, input_dim=384, max_classes=1000):
        super().__init__(input_dim)
        self.net = nn.Linear(input_dim, max_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.label_to_concept = {}
        self.train_x = []
        self.train_y = []
        self.query_x = None
        self.is_trained = False
        
        self.epochs_used = 0
        self.final_train_recall = 0.0
        self.final_test_recall = 0.0
        self.best_test_recall = 0.0

    def ingest(self, vector, label, concept_id):
        self.label_to_concept[label] = concept_id
        self.train_x.append(vector.detach().clone())
        self.train_y.append(label)

    def _train_offline(self):
        if not self.train_x:
            self.is_trained = True
            return
            
        X_train = torch.cat(self.train_x, dim=0)
        Y_train = torch.tensor(self.train_y)
        X_test = self.query_x
        
        max_epochs = 100
        batch_size = 32
        
        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.best_test_recall = 0.0
        self.final_train_recall = 0.0
        self.final_test_recall = 0.0
        
        for epoch in range(1, max_epochs + 1):
            self.net.train()
            epoch_loss = 0.0
            for bx, by in loader:
                self.optimizer.zero_grad()
                logits = self.net(bx)
                loss = self.loss_fn(logits, by)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * bx.size(0)
            epoch_loss /= len(X_train)
            
            # Evaluate train & test recall
            self.net.eval()
            with torch.no_grad():
                train_logits = self.net(X_train)
                seen_labels = list(self.label_to_concept.keys())
                mask = torch.full((train_logits.size(1),), float('-inf'))
                mask[seen_labels] = 0.0
                
                masked_train_logits = train_logits + mask.unsqueeze(0)
                train_preds = torch.argmax(masked_train_logits, dim=1)
                train_correct = (train_preds == Y_train).sum().item()
                train_recall = train_correct / len(X_train)
                
                test_recall = 0.0
                if X_test is not None:
                    test_logits = self.net(X_test)
                    masked_test_logits = test_logits + mask.unsqueeze(0)
                    test_preds = torch.argmax(masked_test_logits, dim=1)
                    test_correct = (test_preds == Y_train).sum().item()
                    test_recall = test_correct / len(X_train)
            
            self.best_test_recall = max(self.best_test_recall, test_recall)
            self.final_train_recall = train_recall
            self.final_test_recall = test_recall
            self.epochs_used = epoch
                
        self.is_trained = True

    def query(self, vector, true_label, true_concept_id):
        if not self.is_trained:
            self._train_offline()
        
        self.net.eval()
        with torch.no_grad():
            logits = self.net(vector).squeeze()
            
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

class OfflineMLP1HiddenClassifier(BaseMemoryModel):
    def __init__(self, input_dim=384, hidden_dim=128, max_classes=1000):
        super().__init__(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.label_to_concept = {}
        self.train_x = []
        self.train_y = []
        self.query_x = None
        self.is_trained = False
        
        self.epochs_used = 0
        self.final_train_recall = 0.0
        self.final_test_recall = 0.0
        self.best_test_recall = 0.0

    def ingest(self, vector, label, concept_id):
        self.label_to_concept[label] = concept_id
        self.train_x.append(vector.detach().clone())
        self.train_y.append(label)

    def _train_offline(self):
        if not self.train_x:
            self.is_trained = True
            return
            
        X_train = torch.cat(self.train_x, dim=0)
        Y_train = torch.tensor(self.train_y)
        X_test = self.query_x
        
        max_epochs = 100
        batch_size = 32
        
        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.best_test_recall = 0.0
        self.final_train_recall = 0.0
        self.final_test_recall = 0.0
        
        for epoch in range(1, max_epochs + 1):
            self.net.train()
            epoch_loss = 0.0
            for bx, by in loader:
                self.optimizer.zero_grad()
                logits = self.net(bx)
                loss = self.loss_fn(logits, by)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * bx.size(0)
            epoch_loss /= len(X_train)
            
            # Evaluate train & test recall
            self.net.eval()
            with torch.no_grad():
                train_logits = self.net(X_train)
                seen_labels = list(self.label_to_concept.keys())
                mask = torch.full((train_logits.size(1),), float('-inf'))
                mask[seen_labels] = 0.0
                
                masked_train_logits = train_logits + mask.unsqueeze(0)
                train_preds = torch.argmax(masked_train_logits, dim=1)
                train_correct = (train_preds == Y_train).sum().item()
                train_recall = train_correct / len(X_train)
                
                test_recall = 0.0
                if X_test is not None:
                    test_logits = self.net(X_test)
                    masked_test_logits = test_logits + mask.unsqueeze(0)
                    test_preds = torch.argmax(masked_test_logits, dim=1)
                    test_correct = (test_preds == Y_train).sum().item()
                    test_recall = test_correct / len(X_train)
            
            self.best_test_recall = max(self.best_test_recall, test_recall)
            self.final_train_recall = train_recall
            self.final_test_recall = test_recall
            self.epochs_used = epoch
                
        self.is_trained = True

    def query(self, vector, true_label, true_concept_id):
        if not self.is_trained:
            self._train_offline()
        
        self.net.eval()
        with torch.no_grad():
            logits = self.net(vector).squeeze()
            
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
# 3. BENCHMARK GENERATOR
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
# METRICS HELPERS
# ==========================================
def format_stats(values, as_pct=True):
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)
    sem = std / math.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * sem
    scale = 100.0 if as_pct else 1.0
    unit = "%" if as_pct else ""
    return f"{mean*scale:.2f}{unit} ± {std*scale:.2f}{unit} (95% CI: {ci95*scale:.2f}{unit})"

def format_percentiles(margins):
    if not margins:
        return "N/A"
    p50, p5, p1 = np.percentile(margins, [50, 5, 1])
    return f"{p50:>5.2f} / {p5:>5.2f} / {p1:>5.2f}"

# ==========================================
# 4. SWEEP 1: CORE COMPARISON
# ==========================================
def run_sweep_core():
    N = 1000
    variance = 0.05
    noise = 0.05
    fpc = 50
    num_seeds = 5
    
    benchmark = MemoryBenchmark(input_dim=384)
    
    print(f"\n{'='*130}")
    print(f"SWEEP 1: GEOMETRY VS PARAMETRIC CAPACITY | N={N} | FPC={fpc} | Var={variance} | Noise={noise} | Seeds={num_seeds}")
    print(f"{'='*130}")
    
    models_to_test = [
        ("Oracle", lambda: OracleRAM(input_dim=384)),
        ("kNN", lambda: UnboundedKNN(input_dim=384)),
        ("OfflineLinear", lambda: OfflineLinearClassifier(input_dim=384, max_classes=N)),
        ("OfflineMLP", lambda: OfflineMLP1HiddenClassifier(input_dim=384, hidden_dim=128, max_classes=N)),
        ("Linear-1Pass", lambda: SinglePassLinearClassifier(input_dim=384, max_classes=N)),
        ("ReplayLinear(200)", lambda: ReplayLinearClassifier(input_dim=384, max_classes=N, buffer_size=200)),
        ("MLP1(128)", lambda: MLP1HiddenClassifier(input_dim=384, hidden_dim=128, max_classes=N)),
        ("ReplayMLP1(128, 200)", lambda: ReplayMLP1HiddenClassifier(input_dim=384, hidden_dim=128, max_classes=N, buffer_size=200))
    ]
    
    for name, model_fn in models_to_test:
        print(f"\nEvaluating Model: {name}")
        recalls = []
        same_c_pcts = []
        diff_c_pcts = []
        global_margins = []
        correct_global_margins = []
        
        offline_epochs = []
        offline_train_recalls = []
        offline_test_recalls = []
        offline_best_test_recalls = []
        
        for seed in range(num_seeds):
            sys.stderr.write(f"\r  Running seed {seed+1}/{num_seeds}...")
            sys.stderr.flush()
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = model_fn()
            train_vecs, labels, concept_ids, query_vecs = benchmark.generate_stream(
                N, facts_per_concept=fpc, variance=variance, noise_level=noise
            )
            
            # Set query_x on offline classifiers
            if hasattr(model, "query_x"):
                model.query_x = query_vecs
                
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
                    global_margins.append(res["margin_global"])
                    
                if res["pred"] == true_lbl:
                    correct += 1
                    if not np.isnan(res["margin_global"]):
                        correct_global_margins.append(res["margin_global"])
                else:
                    seed_errors += 1
                    if res["pred_concept_id"] == true_cid:
                        seed_same_c_errors += 1
                    else:
                        seed_diff_c_errors += 1
            
            recalls.append(correct / N)
            if seed_errors > 0:
                same_c_pcts.append((seed_same_c_errors / seed_errors) * 100)
                diff_c_pcts.append((seed_diff_c_errors / seed_errors) * 100)
            else:
                same_c_pcts.append(0.0)
                diff_c_pcts.append(0.0)
                
            if hasattr(model, "is_trained") and model.is_trained:
                offline_epochs.append(model.epochs_used)
                offline_train_recalls.append(model.final_train_recall)
                offline_test_recalls.append(model.final_test_recall)
                offline_best_test_recalls.append(model.best_test_recall)
        
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()
        
        print(f"  Recall:               {format_stats(recalls, as_pct=True)}")
        print(f"  Same-Concept Error %: {format_stats(same_c_pcts, as_pct=False)} (out of total errors)")
        print(f"  Diff-Concept Error %: {format_stats(diff_c_pcts, as_pct=False)}")
        print(f"  Global Margin (50/5/1%):  {format_percentiles(global_margins)}")
        print(f"  Success Margin (50/5/1%): {format_percentiles(correct_global_margins)}")
        
        if offline_epochs:
            print(f"  [Offline Control Metrics]")
            print(f"    Epochs Used:        {np.mean(offline_epochs):.1f}")
            print(f"    Final Train Recall: {format_stats(offline_train_recalls, as_pct=True)}")
            print(f"    Final Test Recall:  {format_stats(offline_test_recalls, as_pct=True)}")
            print(f"    Best Test Recall:   {format_stats(offline_best_test_recalls, as_pct=True)}")

# ==========================================
# 5. SWEEP 2: RETENTION CURVES
# ==========================================
def run_sweep_retention():
    N = 1000
    variance = 0.05
    noise = 0.05
    fpc = 50
    num_seeds = 5
    checkpoints = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    benchmark = MemoryBenchmark(input_dim=384)
    
    print(f"\n{'='*130}")
    print(f"SWEEP 2: RETENTION CURVES | N={N} | FPC={fpc} | Var={variance} | Noise={noise} | Seeds={num_seeds}")
    print(f"{'='*130}")
    
    models_to_test = [
        ("kNN", lambda: UnboundedKNN(input_dim=384)),
        ("Linear-1Pass", lambda: SinglePassLinearClassifier(input_dim=384, max_classes=N)),
        ("ReplayLinear(200)", lambda: ReplayLinearClassifier(input_dim=384, max_classes=N, buffer_size=200)),
        ("MLP1(128)", lambda: MLP1HiddenClassifier(input_dim=384, hidden_dim=128, max_classes=N)),
        ("ReplayMLP1(128, 200)", lambda: ReplayMLP1HiddenClassifier(input_dim=384, hidden_dim=128, max_classes=N, buffer_size=200))
    ]
    
    # Store results for CSV export
    csv_rows = []
    # Store aggregated curves for plotting
    aggregated_results = {name: {"probe": {cp: [] for cp in checkpoints}, "current": {cp: [] for cp in checkpoints}} for name, _ in models_to_test}
    
    for name, model_fn in models_to_test:
        print(f"Evaluating retention for Model: {name}")
        
        for seed in range(num_seeds):
            sys.stderr.write(f"\r  Running seed {seed+1}/{num_seeds}...")
            sys.stderr.flush()
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = model_fn()
            train_vecs, labels, concept_ids, query_vecs = benchmark.generate_stream(
                N, facts_per_concept=fpc, variance=variance, noise_level=noise
            )
            
            # Streaming training and evaluation at checkpoints
            for cp_idx, cp in enumerate(checkpoints):
                start_i = checkpoints[cp_idx-1] if cp_idx > 0 else 0
                for i in range(start_i, cp):
                    model.ingest(train_vecs[i:i+1], labels[i].item(), concept_ids[i].item())
                
                # Evaluate Probe (First 100 facts seen)
                probe_correct = 0
                for i in range(100):
                    res = model.query(query_vecs[i:i+1], true_label=labels[i].item(), true_concept_id=concept_ids[i].item())
                    if res["pred"] == labels[i].item():
                        probe_correct += 1
                probe_acc = probe_correct / 100.0
                
                # Evaluate Current Stream (Last 100 facts seen)
                current_correct = 0
                for i in range(cp - 100, cp):
                    res = model.query(query_vecs[i:i+1], true_label=labels[i].item(), true_concept_id=concept_ids[i].item())
                    if res["pred"] == labels[i].item():
                        current_correct += 1
                current_acc = current_correct / 100.0
                
                aggregated_results[name]["probe"][cp].append(probe_acc)
                aggregated_results[name]["current"][cp].append(current_acc)
                
                csv_rows.append({
                    "model": name,
                    "checkpoint": cp,
                    "seed": seed,
                    "probe_recall": probe_acc,
                    "current_recall": current_acc
                })
                
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()
        
        # Calculate summary statistics for this model
        probe_final_list = aggregated_results[name]["probe"][1000]
        # Peak probe recall per seed, then mean
        peak_probe_list = []
        for s in range(num_seeds):
            seed_probe_recalls = [aggregated_results[name]["probe"][cp][s] for cp in checkpoints]
            peak_probe_list.append(max(seed_probe_recalls))
            
        retention_ratios = [probe_final / peak if peak > 0 else 0.0 for probe_final, peak in zip(probe_final_list, peak_probe_list)]
        # AURC per seed: average recall across all checkpoints
        aurcs = []
        for s in range(num_seeds):
            seed_probe_recalls = [aggregated_results[name]["probe"][cp][s] for cp in checkpoints]
            aurcs.append(np.mean(seed_probe_recalls))
            
        print(f"  Final Probe Recall: {format_stats(probe_final_list, as_pct=True)}")
        print(f"  Peak Probe Recall:  {format_stats(peak_probe_list, as_pct=True)}")
        print(f"  Retention Ratio:    {format_stats(retention_ratios, as_pct=False)}")
        print(f"  AURC:               {format_stats(aurcs, as_pct=False)}")
        print("-" * 50)
        
    # Export to CSV
    os.makedirs("experiments/results", exist_ok=True)
    csv_file = "experiments/results/diagnostic_telemetry/retention_curves.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "checkpoint", "seed", "probe_recall", "current_recall"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved retention curves data to {csv_file}")
    
    # Generate Matplotlib plots
    plt.figure(figsize=(10, 6))
    colors = {
        "kNN": "#1f77b4",
        "Linear-1Pass": "#ff7f0e",
        "ReplayLinear(200)": "#2ca02c",
        "MLP1(128)": "#d62728",
        "ReplayMLP1(128, 200)": "#9467bd"
    }
    
    for name in colors.keys():
        if name not in aggregated_results:
            continue
        # Compute mean curve
        mean_probe = [np.mean(aggregated_results[name]["probe"][cp]) * 100 for cp in checkpoints]
        mean_current = [np.mean(aggregated_results[name]["current"][cp]) * 100 for cp in checkpoints]
        
        plt.plot(checkpoints, mean_probe, linestyle="-", marker="o", color=colors[name], label=f"{name} (Probe)")
        plt.plot(checkpoints, mean_current, linestyle="--", marker="x", color=colors[name], alpha=0.6, label=f"{name} (Current Stream)")
        
    plt.title("Retention vs. Current Stream Recall over Stream Timeline")
    plt.xlabel("Facts Seen")
    plt.ylabel("Recall Accuracy (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.ylim(-5, 105)
    plt.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, fontsize="small")
    plt.tight_layout()
    plot_file = "experiments/results/diagnostic_telemetry/retention_plots.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Generated retention plots at {plot_file}")

# ==========================================
# 6. SWEEP 3: REPLAY SIZE ABLATION
# ==========================================
def run_sweep_replay():
    N = 1000
    variance = 0.05
    noise = 0.05
    fpc = 50
    num_seeds = 5
    checkpoints = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    benchmark = MemoryBenchmark(input_dim=384)
    
    print(f"\n{'='*130}")
    print(f"SWEEP 3: REPLAY BUFFER ABLATION & EFFICIENCY | N={N} | FPC={fpc} | Var={variance} | Noise={noise} | Seeds={num_seeds}")
    print(f"{'='*130}")
    
    linear_buf_sizes = [0, 50, 100, 200, 500]
    mlp_buf_sizes = [0, 50, 200]
    
    # Track baseline (buffer=0) for gain calculation
    baseline_linear_final_recalls = []
    baseline_mlp_final_recalls = []
    
    # Dictionary to collect results for reporting
    results_linear = {}
    results_mlp = {}
    
    # --- Replay Linear Sweeps ---
    print("\n--- Replay Linear Sweep ---")
    print(f"{'Buffer Size':<12} | {'Final Recall':<30} | {'AURC':<20} | {'Retention Ratio':<20} | {'Gain/Example':<15}")
    print("-" * 105)
    
    for size in linear_buf_sizes:
        final_recalls = []
        aurcs = []
        retention_ratios = []
        
        for seed in range(num_seeds):
            sys.stderr.write(f"\r  Running Linear buffer={size} seed {seed+1}/{num_seeds}...")
            sys.stderr.flush()
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = ReplayLinearClassifier(input_dim=384, max_classes=N, buffer_size=size)
            train_vecs, labels, concept_ids, query_vecs = benchmark.generate_stream(
                N, facts_per_concept=fpc, variance=variance, noise_level=noise
            )
            
            probe_recalls = []
            for cp_idx, cp in enumerate(checkpoints):
                start_i = checkpoints[cp_idx-1] if cp_idx > 0 else 0
                for i in range(start_i, cp):
                    model.ingest(train_vecs[i:i+1], labels[i].item(), concept_ids[i].item())
                
                # evaluate probe
                probe_correct = 0
                for i in range(100):
                    res = model.query(query_vecs[i:i+1], true_label=labels[i].item(), true_concept_id=concept_ids[i].item())
                    if res["pred"] == labels[i].item():
                        probe_correct += 1
                probe_recalls.append(probe_correct / 100.0)
                
            final_recalls.append(probe_recalls[-1])
            aurcs.append(np.mean(probe_recalls))
            peak = max(probe_recalls)
            retention_ratios.append(probe_recalls[-1] / peak if peak > 0 else 0.0)
            
        sys.stderr.write("\r" + " " * 50 + "\r")
        sys.stderr.flush()
        
        if size == 0:
            baseline_linear_final_recalls = final_recalls
            
        # compute gain per stored example
        gains = []
        if size > 0:
            for fr, br in zip(final_recalls, baseline_linear_final_recalls):
                gains.append((fr - br) / size)
            gain_str = f"{np.mean(gains)*100:.4f}%"
        else:
            gain_str = "Baseline"
            
        print(f"{size:<12d} | {format_stats(final_recalls, as_pct=True):<30} | {format_stats(aurcs, as_pct=False):<20} | {format_stats(retention_ratios, as_pct=False):<20} | {gain_str:<15}")
        
    # --- Replay MLP Sweeps ---
    print("\n--- Replay MLP Sweep ---")
    print(f"{'Buffer Size':<12} | {'Final Recall':<30} | {'AURC':<20} | {'Retention Ratio':<20} | {'Gain/Example':<15}")
    print("-" * 105)
    
    for size in mlp_buf_sizes:
        final_recalls = []
        aurcs = []
        retention_ratios = []
        
        for seed in range(num_seeds):
            sys.stderr.write(f"\r  Running MLP buffer={size} seed {seed+1}/{num_seeds}...")
            sys.stderr.flush()
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = ReplayMLP1HiddenClassifier(input_dim=384, hidden_dim=128, max_classes=N, buffer_size=size)
            train_vecs, labels, concept_ids, query_vecs = benchmark.generate_stream(
                N, facts_per_concept=fpc, variance=variance, noise_level=noise
            )
            
            probe_recalls = []
            for cp_idx, cp in enumerate(checkpoints):
                start_i = checkpoints[cp_idx-1] if cp_idx > 0 else 0
                for i in range(start_i, cp):
                    model.ingest(train_vecs[i:i+1], labels[i].item(), concept_ids[i].item())
                
                # evaluate probe
                probe_correct = 0
                for i in range(100):
                    res = model.query(query_vecs[i:i+1], true_label=labels[i].item(), true_concept_id=concept_ids[i].item())
                    if res["pred"] == labels[i].item():
                        probe_correct += 1
                probe_recalls.append(probe_correct / 100.0)
                
            final_recalls.append(probe_recalls[-1])
            aurcs.append(np.mean(probe_recalls))
            peak = max(probe_recalls)
            retention_ratios.append(probe_recalls[-1] / peak if peak > 0 else 0.0)
            
        sys.stderr.write("\r" + " " * 50 + "\r")
        sys.stderr.flush()
        
        if size == 0:
            baseline_mlp_final_recalls = final_recalls
            
        gains = []
        if size > 0:
            for fr, br in zip(final_recalls, baseline_mlp_final_recalls):
                gains.append((fr - br) / size)
            gain_str = f"{np.mean(gains)*100:.4f}%"
        else:
            gain_str = "Baseline"
            
        print(f"{size:<12d} | {format_stats(final_recalls, as_pct=True):<30} | {format_stats(aurcs, as_pct=False):<20} | {format_stats(retention_ratios, as_pct=False):<20} | {gain_str:<15}")

# ==========================================
# 7. SWEEP 4: MLP CAPACITY / WIDTH SWEEP
# ==========================================
def run_sweep_capacity():
    N = 1000
    variance = 0.05
    noise = 0.05
    fpc = 50
    num_seeds = 5
    checkpoints = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    benchmark = MemoryBenchmark(input_dim=384)
    
    print(f"\n{'='*130}")
    print(f"SWEEP 4: MLP CAPACITY & WIDTH SWEEP | N={N} | FPC={fpc} | Var={variance} | Noise={noise} | Seeds={num_seeds}")
    print(f"{'='*130}")
    
    widths = [64, 128, 256, 512]
    
    print(f"{'Width':<12} | {'Final Recall':<30} | {'AURC':<20} | {'Retention Ratio':<20}")
    print("-" * 90)
    
    for w in widths:
        final_recalls = []
        aurcs = []
        retention_ratios = []
        
        for seed in range(num_seeds):
            sys.stderr.write(f"\r  Running Width={w} seed {seed+1}/{num_seeds}...")
            sys.stderr.flush()
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = MLP1HiddenClassifier(input_dim=384, hidden_dim=w, max_classes=N)
            train_vecs, labels, concept_ids, query_vecs = benchmark.generate_stream(
                N, facts_per_concept=fpc, variance=variance, noise_level=noise
            )
            
            probe_recalls = []
            for cp_idx, cp in enumerate(checkpoints):
                start_i = checkpoints[cp_idx-1] if cp_idx > 0 else 0
                for i in range(start_i, cp):
                    model.ingest(train_vecs[i:i+1], labels[i].item(), concept_ids[i].item())
                
                # evaluate probe
                probe_correct = 0
                for i in range(100):
                    res = model.query(query_vecs[i:i+1], true_label=labels[i].item(), true_concept_id=concept_ids[i].item())
                    if res["pred"] == labels[i].item():
                        probe_correct += 1
                probe_recalls.append(probe_correct / 100.0)
                
            final_recalls.append(probe_recalls[-1])
            aurcs.append(np.mean(probe_recalls))
            peak = max(probe_recalls)
            retention_ratios.append(probe_recalls[-1] / peak if peak > 0 else 0.0)
            
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()
        
        print(f"{w:<12d} | {format_stats(final_recalls, as_pct=True):<30} | {format_stats(aurcs, as_pct=False):<20} | {format_stats(retention_ratios, as_pct=False):<20}")

# ==========================================
# 8. SWEEP 5: DIMENSION × DENSITY MATRIX
# ==========================================
def run_sweep_geometry():
    N = 1000
    variance = 0.05
    noise = 0.05
    num_seeds = 10
    
    dims = [64, 128, 256, 384, 768]
    densities = [10, 50, 100, 200]
    
    print(f"\n{'='*130}")
    print(f"SWEEP 5: GEOMETRIC CAPACITY MATRIX (kNN STRESS TEST) | N={N} | Var={variance} | Noise={noise} | Seeds={num_seeds}")
    print(f"{'='*130}")
    
    # Matrices to hold aggregated results for plotting
    matrix_recall = np.zeros((len(dims), len(densities)))
    matrix_margin = np.zeros((len(dims), len(densities)))
    matrix_same_c = np.zeros((len(dims), len(densities)))
    
    # Detailed output logging
    for d_idx, dim in enumerate(dims):
        print(f"\nDimension: {dim}")
        print(f"  {'Facts/Concept':<15} | {'Recall':<25} | {'P1 Success Margin':<20} | {'Same-Concept Error %':<20}")
        print("  " + "-" * 88)
        
        benchmark = MemoryBenchmark(input_dim=dim)
        
        for f_idx, fpc in enumerate(densities):
            recalls = []
            all_correct_margins = []
            seed_same_c_pcts = []
            
            for seed in range(num_seeds):
                sys.stderr.write(f"\r    Running fpc={fpc} seed {seed+1}/{num_seeds}...")
                sys.stderr.flush()
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                model = UnboundedKNN(input_dim=dim)
                train_vecs, labels, concept_ids, query_vecs = benchmark.generate_stream(
                    N, facts_per_concept=fpc, variance=variance, noise_level=noise
                )
                
                for i in range(N):
                    model.ingest(train_vecs[i:i+1], labels[i].item(), concept_ids[i].item())
                    
                correct = 0
                errors = 0
                same_c_errors = 0
                
                for i in range(N):
                    true_lbl = labels[i].item()
                    true_cid = concept_ids[i].item()
                    res = model.query(query_vecs[i:i+1], true_label=true_lbl, true_concept_id=true_cid)
                    
                    if res["pred"] == true_lbl:
                        correct += 1
                        if not np.isnan(res["margin_global"]):
                            all_correct_margins.append(res["margin_global"])
                    else:
                        errors += 1
                        if res["pred_concept_id"] == true_cid:
                            same_c_errors += 1
                
                recalls.append(correct / N)
                if errors > 0:
                    seed_same_c_pcts.append((same_c_errors / errors) * 100)
                else:
                    seed_same_c_pcts.append(0.0)
                    
            sys.stderr.write("\r" + " " * 40 + "\r")
            sys.stderr.flush()
            
            mean_r = np.mean(recalls)
            p1_margin = np.percentile(all_correct_margins, 1) if all_correct_margins else 0.0
            mean_same_c = np.mean(seed_same_c_pcts)
            
            matrix_recall[d_idx, f_idx] = mean_r
            matrix_margin[d_idx, f_idx] = p1_margin
            matrix_same_c[d_idx, f_idx] = mean_same_c
            
            p1_str = f"{p1_margin:.4f}" if all_correct_margins else "N/A"
            print(f"  {fpc:<15d} | {format_stats(recalls, as_pct=True):<25} | {p1_str:<20} | {mean_same_c:.2f}%")
            
    # Generate Dual Heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Recall Heatmap
    im1 = axs[0].imshow(matrix_recall * 100, cmap="viridis", aspect="auto", origin="lower", vmin=0, vmax=100)
    axs[0].set_title("kNN Recall Accuracy (%)")
    axs[0].set_xticks(np.arange(len(densities)))
    axs[0].set_xticklabels(densities)
    axs[0].set_yticks(np.arange(len(dims)))
    axs[0].set_yticklabels(dims)
    axs[0].set_xlabel("Facts per Concept (Density)")
    axs[0].set_ylabel("Representation Dimension")
    fig.colorbar(im1, ax=axs[0], label="Recall (%)")
    
    # Annotate Recall cells
    for i in range(len(dims)):
        for j in range(len(densities)):
            val = matrix_recall[i, j] * 100
            axs[0].text(j, i, f"{val:.1f}%", ha="center", va="center", color="white" if val < 50 else "black", fontweight="bold")
            
    # 2. P1 Margin Heatmap
    im2 = axs[1].imshow(matrix_margin, cmap="plasma", aspect="auto", origin="lower")
    axs[1].set_title("kNN P1 Success Margin")
    axs[1].set_xticks(np.arange(len(densities)))
    axs[1].set_xticklabels(densities)
    axs[1].set_yticks(np.arange(len(dims)))
    axs[1].set_yticklabels(dims)
    axs[1].set_xlabel("Facts per Concept (Density)")
    axs[1].set_ylabel("Representation Dimension")
    fig.colorbar(im2, ax=axs[1], label="P1 Success Margin")
    
    # Annotate Margin cells
    for i in range(len(dims)):
        for j in range(len(densities)):
            val = matrix_margin[i, j]
            axs[1].text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < np.mean(matrix_margin) else "black", fontweight="bold")
            
    plt.tight_layout()
    heatmap_file = "experiments/results/diagnostic_telemetry/geometry_heatmap.png"
    plt.savefig(heatmap_file, dpi=150)
    plt.close()
    print(f"\nSaved geometric heatmaps to {heatmap_file}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Parametric Isolation & Forgetting Study")
    parser.add_argument(
        "--sweep", 
        type=str, 
        default="all", 
        choices=["all", "core", "retention", "replay", "capacity", "geometry"],
        help="Which sweep to run (default: all)"
    )
    args = parser.parse_args()
    
    log_dir = "experiments/results/diagnostic_telemetry/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure dual logger
    sys.stdout = DualWriter(os.path.join(log_dir, "parametric_study.log"))
    
    start_time = time.time()
    
    print("="*130)
    print(f"PARAMETRIC ISOLATION & FORGETTING STUDY | RUN TYPE: {args.sweep.upper()}")
    print("="*130)
    
    if args.sweep in ["all", "core"]:
        run_sweep_core()
    if args.sweep in ["all", "retention"]:
        run_sweep_retention()
    if args.sweep in ["all", "replay"]:
        run_sweep_replay()
    if args.sweep in ["all", "capacity"]:
        run_sweep_capacity()
    if args.sweep in ["all", "geometry"]:
        run_sweep_geometry()
        
    print(f"\n{'='*130}")
    print(f"PARAMETRIC STUDY COMPLETED | Total Runtime: {(time.time() - start_time)/60:.2f} minutes.")
    print("="*130)

if __name__ == "__main__":
    main()
