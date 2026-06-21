import sys
import os
import csv
import torch
import torch.nn as nn
import numpy as np
import random

# Add mnc_project to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mnc_project'))

from mnc.layers import MNCLinear
from pipeline import JournalPipeline

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class ScaleDistances(nn.Module):
    def __init__(self, shift, scale):
        super().__init__()
        self.shift = shift
        self.scale = scale
    def forward(self, x):
        return (x + self.shift) / self.scale

def autocalibrate_scale_distances(in_features, num_samples=1000):
    set_seed(42)
    x = torch.randn(num_samples, in_features)
    x = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8)
    w = torch.randn(num_samples, in_features)
    w = w / (w.norm(p=2, dim=1, keepdim=True) + 1e-8)
    neg_l1 = -torch.norm(x - w, p=1, dim=1)
    mean_val = neg_l1.mean().item()
    std_val = neg_l1.std().item()
    shift = -mean_val
    scale = 2.0 * std_val if std_val > 1e-5 else 1.0
    return shift, scale

SHIFT0, SCALE0 = autocalibrate_scale_distances(384)
SHIFT1, SCALE1 = autocalibrate_scale_distances(32)

FACTS = [
    {"statement": "The blue folder is in the third drawer.", "label": 0},
    {"statement": "The access code for the main server is 7734.", "label": 1},
    {"statement": "Sarah's strategy meeting was moved to Tuesday at 3 PM.", "label": 2},
    {"statement": "The backup generator requires unleaded fuel.", "label": 3},
    {"statement": "Sector 4 security cameras undergo maintenance at midnight.", "label": 4}
]

INTERFERENCE = [
    "The red folder is resting on the top desk.",
    "Someone left a coffee mug in the breakroom.",
    "The access code for the guest wifi is 9912.",
    "It is raining heavily outside today."
]

JOURNAL_LINES = []
for f in FACTS:
    JOURNAL_LINES.append({"text": f["statement"], "label": f["label"]})
for idx, text in enumerate(INTERFERENCE):
    JOURNAL_LINES.append({"text": text, "label": 5 + idx})

def cosine_similarity(v1, v2):
    dot = torch.sum(v1 * v2)
    n1 = torch.norm(v1, p=2)
    n2 = torch.norm(v2, p=2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return (dot / (n1 * n2)).item()

def run_parameter_overlap(seed=42):
    set_seed(seed)
    pipeline = JournalPipeline()
    
    # 1. Instantiate the model
    model = nn.Sequential(
        MNCLinear(384, 32),
        ScaleDistances(SHIFT0, SCALE0),
        nn.Tanh(),
        MNCLinear(32, 10),
        ScaleDistances(SHIFT1, SCALE1)
    )
    loss_fn = nn.CrossEntropyLoss()
    
    # 2. Extract gradients for each individual item independently
    gradients_l0 = {}
    gradients_l3 = {}
    
    for i, item in enumerate(JOURNAL_LINES):
        model.zero_grad()
        vec = pipeline.embed_sentence(item["text"])
        target = torch.tensor([item["label"]])
        
        logits = model(vec)
        loss = loss_fn(logits, target)
        loss.backward()
        
        # Save flat copies of the gradients
        gradients_l0[i] = torch.clone(model[0].W.grad.data.view(-1))
        gradients_l3[i] = torch.clone(model[3].W.grad.data.view(-1))
        
    # 3. Compute the similarity matrices
    num_items = len(JOURNAL_LINES)
    matrix_l0 = np.zeros((num_items, num_items))
    matrix_l3 = np.zeros((num_items, num_items))
    
    for r in range(num_items):
        for c in range(num_items):
            matrix_l0[r, c] = cosine_similarity(gradients_l0[r], gradients_l0[c])
            matrix_l3[r, c] = cosine_similarity(gradients_l3[r], gradients_l3[c])
            
    # 4. Save results to CSV
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'parameter_overlap.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "row_task", "col_task", "cosine_similarity"])
        for r in range(num_items):
            for c in range(num_items):
                writer.writerow(["Layer 0 (Bottleneck)", r, c, matrix_l0[r, c]])
                writer.writerow(["Layer 2 (Templates)", r, c, matrix_l3[r, c]])
                
    print(f"\n[SUCCESS] Gradient overlap matrices saved to: {csv_path}")
    
    # 5. Print analysis summary
    # Off-diagonal elements (excluding self-similarity)
    mask = ~np.eye(num_items, dtype=bool)
    off_diag_l0 = matrix_l0[mask]
    off_diag_l3 = matrix_l3[mask]
    
    print("\n=== Parameter Overlap Diagnostic Analysis ===")
    print(f"Layer 0 (Bottleneck 384->32) -> Average Cross-Task Overlap: {off_diag_l0.mean():.4f} (Max: {off_diag_l0.max():.4f})")
    print(f"Layer 2 (Templates 32->10)   -> Average Cross-Task Overlap: {off_diag_l3.mean():.4f} (Max: {off_diag_l3.max():.4f})")
    
    print("\nVisual representation of Layer 0 (Bottleneck) Cosine Similarity:")
    for r in range(num_items):
        row_str = " ".join([f"{matrix_l0[r, c]:+.2f}" for c in range(num_items)])
        print(f"Task {r} | {row_str}")
        
    print("\nVisual representation of Layer 2 (Templates) Cosine Similarity:")
    for r in range(num_items):
        row_str = " ".join([f"{matrix_l3[r, c]:+.2f}" for c in range(num_items)])
        print(f"Task {r} | {row_str}")

if __name__ == "__main__":
    run_parameter_overlap()
