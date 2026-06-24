import sys
import os
import csv
import time
import torch
import torch.nn as nn
import numpy as np
import random

# Add mnc_project to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mnc_project'))

from mnc.layers import MNCLinear
from mnc.memory import MESUEngine
from pipeline import JournalPipeline

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# Autocalibration for 384 input features (static)
SHIFT_384, SCALE_384 = autocalibrate_scale_distances(384)

def generate_facts(N):
    facts = []
    random_gen = random.Random(12345)
    locations = ["drawer", "cabinet", "locker", "safe", "box", "desk", "closet", "shelf", "bin", "basket"]
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "black", "white", "gray"]
    
    for i in range(N):
        color = random_gen.choice(colors)
        loc = random_gen.choice(locations)
        num = random_gen.randint(1000, 9999)
        
        template_type = i % 3
        if template_type == 0:
            statement = f"The access code for the main server room {i} is {num}."
            query = f"What is the access code for server room {i}?"
        elif template_type == 1:
            statement = f"The {color} folder for project {i} is in the {loc}."
            query = f"Where is the {color} folder for project {i} kept?"
        else:
            statement = f"Sarah's meeting for task {i} is scheduled on Tuesday at {i % 12 + 1} PM."
            query = f"When is the meeting for task {i} scheduled?"
            
        facts.append({
            "statement": statement,
            "query": query,
            "label": i
        })
    return facts

def evaluate_fact(model, fact, embedded_query):
    model.eval()
    with torch.no_grad():
        logits = model(embedded_query)
        pred = logits.argmax(dim=-1).item()
    return 1.0 if pred == fact["label"] else 0.0

def run_width_evaluation(width, num_facts, seeds, pipeline, facts, embedded_statements, embedded_queries, shift_w, scale_w, u2_enabled):
    results = []
    for seed in seeds:
        set_seed(seed)
        
        model = nn.Sequential(
            MNCLinear(384, width),
            ScaleDistances(SHIFT_384, SCALE_384),
            nn.Tanh(),
            MNCLinear(width, num_facts),
            ScaleDistances(shift_w, scale_w)
        )
        
        engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=0.02, u2_enabled=u2_enabled)
        loss_fn = nn.CrossEntropyLoss()
        
        # Train sequentially
        for idx in range(num_facts):
            statement_vec = embedded_statements[idx]
            target = torch.tensor([idx])
            
            model.train()
            for _ in range(15):
                noise = torch.randn_like(statement_vec) * 0.05
                noisy_vec = statement_vec + noise
                logits = model(noisy_vec)
                loss = loss_fn(logits, target)
                loss.backward()
                engine.step(loss.item())
                engine.zero_grad()
            
        # Batched final recall evaluation
        model.eval()
        with torch.no_grad():
            qvecs = torch.cat(embedded_queries, dim=0)
            logits = model(qvecs)
            preds = logits.argmax(dim=-1)
            targets = torch.tensor([f["label"] for f in facts])
            final_recalls = (preds == targets).float().tolist()
            
        mean_final = np.mean(final_recalls)
        results.append(mean_final)
    return np.mean(results)

def fit_scaling_law(n_vals, recall_vals):
    x = np.log(np.array(n_vals))
    eps = 1e-5
    y = np.log(np.array(recall_vals) + eps)
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -slope
    A = np.exp(intercept)
    return A, alpha

def main():
    print("====================================================")
    print("   MESU u2-Ablation Micro-Sweep (Baseline Width 32) ")
    print("====================================================")
    
    pipeline = JournalPipeline()
    seeds = [42, 101, 202]
    width = 32
    capacity_steps = [5, 10, 20, 50, 100, 200]
    
    shift_w, scale_w = autocalibrate_scale_distances(width)
    
    print("\n[1/2] Running with u2 ENABLED (MESU default)...")
    recalls_u2 = []
    for n in capacity_steps:
        facts = generate_facts(n)
        embedded_statements = [pipeline.embed_sentence(f["statement"]) for f in facts]
        embedded_queries = [pipeline.embed_sentence(f["query"]) for f in facts]
        mean_rec = run_width_evaluation(width, n, seeds, pipeline, facts, embedded_statements, embedded_queries, shift_w, scale_w, u2_enabled=True)
        recalls_u2.append(mean_rec)
        print(f"  N = {n:3d} | Recall: {mean_rec*100:5.2f}%")
        
    A_u2, alpha_u2 = fit_scaling_law(capacity_steps, recalls_u2)
    print(f"  Fitted u2 Enabled: Recall(N) = {A_u2:.4f} / N^{alpha_u2:.4f}")
    
    print("\n[2/2] Running with u2 ABLATED (u2_enabled=False)...")
    recalls_no_u2 = []
    for n in capacity_steps:
        facts = generate_facts(n)
        embedded_statements = [pipeline.embed_sentence(f["statement"]) for f in facts]
        embedded_queries = [pipeline.embed_sentence(f["query"]) for f in facts]
        mean_rec = run_width_evaluation(width, n, seeds, pipeline, facts, embedded_statements, embedded_queries, shift_w, scale_w, u2_enabled=False)
        recalls_no_u2.append(mean_rec)
        print(f"  N = {n:3d} | Recall: {mean_rec*100:5.2f}%")
        
    A_no_u2, alpha_no_u2 = fit_scaling_law(capacity_steps, recalls_no_u2)
    print(f"  Fitted u2 Ablated: Recall(N) = {A_no_u2:.4f} / N^{alpha_no_u2:.4f}")
    
    print("\n====================================================")
    print("   Scientific Comparison")
    print("====================================================")
    print(f"  MESU (u2 Enabled): alpha = {alpha_u2:.4f}")
    print(f"  MESU (u2 Ablated): alpha = {alpha_no_u2:.4f}")
    print(f"  Difference:       {alpha_no_u2 - alpha_u2:+.4f}")
    
    if alpha_no_u2 > alpha_u2 + 0.15:
        print("  Verdict: The slow consolidation pathway (u2) significantly mitigates the interference decay exponent.")
    else:
        print("  Verdict: The u2 pathway has negligible impact on the power-law decay exponent.")

if __name__ == "__main__":
    main()
