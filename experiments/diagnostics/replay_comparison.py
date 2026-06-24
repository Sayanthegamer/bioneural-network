import sys
import os
import csv
import time
import torch
import torch.nn as nn
import numpy as np
import random

# Add mnc_project to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'mnc_project'))

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

def run_replay_experiment(method, buffer_size, seeds, csv_path, num_facts=50):
    print(f"\n[*] Evaluating: Method={method} | Replay Buffer Size={buffer_size}")
    
    pipeline = JournalPipeline()
    facts = generate_facts(num_facts)
    
    # Pre-embed
    embedded_statements = [pipeline.embed_sentence(f["statement"]) for f in facts]
    embedded_queries = [pipeline.embed_sentence(f["query"]) for f in facts]
    
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
        
        loss_fn = nn.CrossEntropyLoss()
        
        if method == "MESU":
            engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=0.02)
            optimizer = None
        else:
            engine = None
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            
        replay_buffer = []
        initial_recalls = []
        
        # Sequential Ingestion Loop
        for idx in range(num_facts):
            statement_vec = embedded_statements[idx]
            target = torch.tensor([idx])
            
            # Train for 15 steps
            for _ in range(15):
                model.train()
                # 1. Update on the current fact
                noise = torch.randn_like(statement_vec) * 0.05
                noisy_vec = statement_vec + noise
                logits = model(noisy_vec)
                loss = loss_fn(logits, target)
                loss.backward()
                
                # 2. Accumulate/Replay update from the buffer if applicable
                if buffer_size > 0 and len(replay_buffer) > 0:
                    # Sample a random item from the buffer
                    replayed_item = random.choice(replay_buffer)
                    rep_vec = replayed_item["vec"]
                    rep_target = replayed_item["target"]
                    
                    rep_noise = torch.randn_like(rep_vec) * 0.05
                    rep_noisy_vec = rep_vec + rep_noise
                    rep_logits = model(rep_noisy_vec)
                    rep_loss = loss_fn(rep_logits, rep_target)
                    rep_loss.backward()
                    
                # 3. Step optimizer
                if engine is not None:
                    engine.step(loss.item())
                    engine.zero_grad()
                else:
                    optimizer.step()
                    # Sphere projection for SGD
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if "W" in name and param.data.dim() == 2:
                                param.data.copy_(param.data / (param.data.norm(p=2, dim=1, keepdim=True) + 1e-8))
                    optimizer.zero_grad()
            
            # Save the immediate recall on this fact
            acc_init = evaluate_fact(model, facts[idx], embedded_queries[idx])
            initial_recalls.append(acc_init)
            
            # Add to replay buffer (FIFO eviction)
            if buffer_size > 0:
                replay_buffer.append({"vec": statement_vec, "target": target})
                if len(replay_buffer) > buffer_size:
                    replay_buffer.pop(0) # Evict oldest
                    
        # Final recall evaluation
        final_recalls = []
        for idx in range(num_facts):
            acc_final = evaluate_fact(model, facts[idx], embedded_queries[idx])
            final_recalls.append(acc_final)
            
        mean_initial = np.mean(initial_recalls)
        mean_final = np.mean(final_recalls)
        forgetting = mean_initial - mean_final
        
        # Calculate template drift
        drift_0 = torch.norm(model[0].W.data - W0_layer0, p=2).item()
        drift_3 = torch.norm(model[3].W.data - W0_layer3, p=2).item()
        mean_drift = (drift_0 + drift_3) / 2.0
        
        # Variance statistics (or empty if SGD)
        if engine is not None:
            var_all = torch.cat([v.view(-1) for v in engine.variances.values()])
            v_mean = var_all.mean().item()
            v_min = var_all.min().item()
            v_max = var_all.max().item()
        else:
            v_mean, v_min, v_max = 0.0, 0.0, 0.0
            
        runtime = time.time() - t_start
        
        # Recall per stored item (Memory Efficiency)
        # Stored items is the buffer size capacity, using max(1, buffer_size) to avoid div-by-zero
        efficiency = mean_final / max(1, buffer_size)
        
        res_row = {
            "seed": seed,
            "experiment": "replay_comparison",
            "config": f"{method}_replay={buffer_size}",
            "alpha_decay": 0.02 if method == "MESU" else "",
            "u2_enabled": True if method == "MESU" else "",
            "replay_buffer_size": buffer_size,
            "num_facts": num_facts,
            "recall": mean_final,
            "forgetting": forgetting,
            "drift": mean_drift,
            "variance_mean": v_mean,
            "variance_min": v_min,
            "variance_max": v_max,
            "runtime_sec": runtime,
            "efficiency": efficiency
        }
        results.append(res_row)
        print(f"  Seed {seed:3} | Recall: {mean_final*100:5.1f}% | Forgetting: {forgetting*100:5.1f}% | Efficiency: {efficiency:.4f} | Drift: {mean_drift:.4f}")
        
    # Save results to global CSV
    file_exists = os.path.exists(csv_path)
    headers = [
        "seed", "experiment", "config", "alpha_decay", "u2_enabled", "replay_buffer_size", "num_facts",
        "recall", "forgetting", "drift", "variance_mean", "variance_min", "variance_max", "runtime_sec", "efficiency"
    ]
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Calculate and return aggregate statistics
    rec_all = [r["recall"] for r in results]
    forg_all = [r["forgetting"] for r in results]
    eff_all = [r["efficiency"] for r in results]
    
    return np.mean(rec_all), np.std(rec_all), np.mean(forg_all), np.std(forg_all), np.mean(eff_all), np.std(eff_all)

if __name__ == "__main__":
    seeds = [42, 101, 202, 303, 404]
    csv_path = "experiments/results/diagnostic_telemetry/replay_comparison.csv"
    
    # Remove existing replay_comparison.csv if it exists to start fresh
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    eval_matrix = [
        ("SGD", 0),
        ("SGD", 10),
        ("SGD", 50),
        ("SGD", 100),
        ("MESU", 0),
        ("MESU", 10),
        ("MESU", 50)
    ]
    
    summary_stats = []
    
    print("====================================================")
    print("  MESU vs SGD Experience Replay Benchmark")
    print("====================================================")
    
    for method, buf_size in eval_matrix:
        m_rec, s_rec, m_forg, s_forg, m_eff, s_eff = run_replay_experiment(method, buf_size, seeds, csv_path)
        summary_stats.append((method, buf_size, m_rec, s_rec, m_forg, s_forg, m_eff, s_eff))
        
    print("\n====================================================")
    print("  MESU vs SGD Experience Replay Summary Table")
    print("====================================================")
    print(f"{'Method':<6} | {'Buffer':<6} | {'Recall (Mean +/- Std)':<25} | {'Forgetting (Mean +/- Std)':<25} | {'Efficiency (Mean +/- Std)':<25}")
    print("-" * 105)
    for method, buf_size, m_rec, s_rec, m_forg, s_forg, m_eff, s_eff in summary_stats:
        print(f"{method:<6} | {buf_size:<6} | {m_rec*100:5.2f}% +/- {s_rec*100:5.2f}% | {m_forg*100:5.2f}% +/- {s_forg*100:5.2f}% | {m_eff:.4f} +/- {s_eff:.4f}")
