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

SHIFT0, SCALE0 = autocalibrate_scale_distances(384)
SHIFT1, SCALE1 = autocalibrate_scale_distances(32)

def generate_facts(N):
    """Generates N unique facts and queries using templates to avoid semantic collision."""
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

def evaluate_fact(model, pipeline, fact, embedded_query=None):
    model.eval()
    with torch.no_grad():
        if embedded_query is not None:
            qvec = embedded_query
        else:
            qvec = pipeline.embed_sentence(fact["query"])
        logits = model(qvec)
        pred = logits.argmax(dim=-1).item()
    return 1.0 if pred == fact["label"] else 0.0

def run_capacity_evaluation(num_facts, seeds, csv_path):
    print(f"\n====================================================")
    print(f"  MESU Capacity Wall Scaling Sweep: N = {num_facts}")
    print(f"====================================================")
    
    pipeline = JournalPipeline()
    
    # Generate all facts
    facts = generate_facts(num_facts)
    
    # Pre-embed statements and queries for speed
    print(f"[*] Pre-embedding {num_facts} facts and queries...")
    embedded_statements = [pipeline.embed_sentence(f["statement"]) for f in facts]
    embedded_queries = [pipeline.embed_sentence(f["query"]) for f in facts]
    print("[*] Embedding complete.")
    
    results = []
    
    for seed in seeds:
        t_start = time.time()
        set_seed(seed)
        
        model = nn.Sequential(
            MNCLinear(384, 32),
            ScaleDistances(SHIFT0, SCALE0),
            nn.Tanh(),
            MNCLinear(32, num_facts),
            ScaleDistances(SHIFT1, SCALE1)
        )
        
        W0_layer0 = torch.clone(model[0].W.data)
        W0_layer3 = torch.clone(model[3].W.data)
        
        engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=0.02)
        loss_fn = nn.CrossEntropyLoss()
        
        initial_recalls = []
        
        # Train sequentially on facts
        for idx in range(num_facts):
            statement_vec = embedded_statements[idx]
            target = torch.tensor([idx])
            
            # Train for 15 steps
            model.train()
            for _ in range(15):
                noise = torch.randn_like(statement_vec) * 0.05
                noisy_vec = statement_vec + noise
                logits = model(noisy_vec)
                loss = loss_fn(logits, target)
                loss.backward()
                engine.step(loss.item())
                engine.zero_grad()
                
            # Measure immediate recall baseline
            acc_init = evaluate_fact(model, pipeline, facts[idx], embedded_query=embedded_queries[idx])
            initial_recalls.append(acc_init)
            
        # Final recall evaluation across all facts
        final_recalls = []
        for idx in range(num_facts):
            acc_final = evaluate_fact(model, pipeline, facts[idx], embedded_query=embedded_queries[idx])
            final_recalls.append(acc_final)
            
        mean_initial = np.mean(initial_recalls)
        mean_final = np.mean(final_recalls)
        forgetting = mean_initial - mean_final
        
        # Calculate drift
        drift_0 = torch.norm(model[0].W.data - W0_layer0, p=2).item()
        drift_3 = torch.norm(model[3].W.data - W0_layer3, p=2).item()
        mean_drift = (drift_0 + drift_3) / 2.0
        
        # Variance statistics
        var_all = torch.cat([v.view(-1) for v in engine.variances.values()])
        
        runtime = time.time() - t_start
        
        res_row = {
            "seed": seed,
            "experiment": "capacity_wall",
            "config": f"num_facts={num_facts}",
            "alpha_decay": 0.02,
            "u2_enabled": True,
            "replay_buffer_size": 0,
            "num_facts": num_facts,
            "recall": mean_final,
            "forgetting": forgetting,
            "drift": mean_drift,
            "variance_mean": var_all.mean().item(),
            "variance_min": var_all.min().item(),
            "variance_max": var_all.max().item(),
            "runtime_sec": runtime
        }
        results.append(res_row)
        print(f"  Seed {seed:3} | Final Recall: {mean_final*100:.1f}% | Forgetting: {forgetting*100:.1f}% | Drift: {mean_drift:.4f} | Runtime: {runtime:.2f}s")
        
    # Append results to global CSV
    file_exists = os.path.exists(csv_path)
    headers = [
        "seed", "experiment", "config", "alpha_decay", "u2_enabled", "replay_buffer_size", "num_facts",
        "recall", "forgetting", "drift", "variance_mean", "variance_min", "variance_max", "runtime_sec"
    ]
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Calculate stats for reporting
    final_recalls_all = [r["recall"] for r in results]
    forgettings_all = [r["forgetting"] for r in results]
    
    mean_rec = np.mean(final_recalls_all)
    std_rec = np.std(final_recalls_all)
    mean_forg = np.mean(forgettings_all)
    std_forg = np.std(forgettings_all)
    
    # Identify retention band
    if mean_rec >= 0.80:
        band = "Perfect Retention (>= 80%)"
    elif mean_rec >= 0.60:
        band = "Moderate Retention (>= 60%)"
    elif mean_rec >= 0.50:
        band = "Minimal Retention (>= 50%)"
    else:
        band = "Failed Retention (< 50%)"
        
    print(f"--> [SUMMARY] N = {num_facts} | Recall: {mean_rec*100:.2f}% +/- {std_rec*100:.2f}% | Forgetting: {mean_forg*100:.2f}% +/- {std_forg*100:.2f}%")
    print(f"    Retention Band: {band}")
    return mean_rec, std_rec, mean_forg, std_forg

if __name__ == "__main__":
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    csv_path = "experiments/results/capacity_wall.csv"
    
    # Remove existing capacity_wall.csv if it exists to start fresh
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    capacity_steps = [5, 10, 20, 50, 100, 200, 400, 800]
    
    summary_stats = []
    for step in capacity_steps:
        m_rec, s_rec, m_forg, s_forg = run_capacity_evaluation(step, seeds, csv_path)
        summary_stats.append((step, m_rec, s_rec, m_forg, s_forg))
        
    print("\n====================================================")
    print("  MESU Capacity Scaling Summary Table")
    print("====================================================")
    print(f"{'N Facts':<10} | {'Recall (Mean +/- Std)':<25} | {'Forgetting (Mean +/- Std)':<25} | {'Retention Band':<25}")
    print("-" * 92)
    for step, m_rec, s_rec, m_forg, s_forg in summary_stats:
        if m_rec >= 0.80:
            band = "Perfect (>= 80%)"
        elif m_rec >= 0.60:
            band = "Moderate (>= 60%)"
        elif m_rec >= 0.50:
            band = "Minimal (>= 50%)"
        else:
            band = "Failed (< 50%)"
        print(f"{step:<10} | {m_rec*100:5.2f}% +/- {s_rec*100:5.2f}% | {m_forg*100:5.2f}% +/- {s_forg*100:5.2f}% | {band:<25}")
