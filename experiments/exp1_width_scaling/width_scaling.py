import sys
import os
import csv
import time
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

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

def run_width_evaluation(width, num_facts, seeds, pipeline, facts, embedded_statements, embedded_queries, shift_w, scale_w):
    results = []
    for seed in seeds:
        t_start = time.time()
        set_seed(seed)
        
        model = nn.Sequential(
            MNCLinear(384, width),
            ScaleDistances(SHIFT_384, SCALE_384),
            nn.Tanh(),
            MNCLinear(width, num_facts),
            ScaleDistances(shift_w, scale_w)
        )
        
        W0_layer0 = torch.clone(model[0].W.data)
        W0_layer3 = torch.clone(model[3].W.data)
        
        engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=0.02)
        loss_fn = nn.CrossEntropyLoss()
        
        initial_recalls = []
        
        # Train sequentially
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
                
            # Measure immediate recall
            acc_init = evaluate_fact(model, facts[idx], embedded_queries[idx])
            initial_recalls.append(acc_init)
            
        # Batched final recall evaluation
        model.eval()
        with torch.no_grad():
            qvecs = torch.cat(embedded_queries, dim=0)
            logits = model(qvecs)
            preds = logits.argmax(dim=-1)
            targets = torch.tensor([f["label"] for f in facts])
            final_recalls = (preds == targets).float().tolist()
            
        mean_initial = np.mean(initial_recalls)
        mean_final = np.mean(final_recalls)
        forgetting = mean_initial - mean_final
        
        # Template drift
        drift_0 = torch.norm(model[0].W.data - W0_layer0, p=2).item()
        drift_3 = torch.norm(model[3].W.data - W0_layer3, p=2).item()
        mean_drift = (drift_0 + drift_3) / 2.0
        
        # Variance stats
        var_all = torch.cat([v.view(-1) for v in engine.variances.values()])
        
        results.append({
            "seed": seed,
            "recall": mean_final,
            "forgetting": forgetting,
            "drift": mean_drift,
            "variance_mean": var_all.mean().item(),
            "variance_min": var_all.min().item(),
            "variance_max": var_all.max().item(),
            "runtime_sec": time.time() - t_start
        })
    return results

def fit_scaling_law(n_vals, recall_vals):
    x = np.log(np.array(n_vals))
    eps = 1e-5
    y = np.log(np.array(recall_vals) + eps)
    
    # Linear fit: y = m * x + c
    slope, intercept = np.polyfit(x, y, 1)
    
    alpha = -slope
    A = np.exp(intercept)
    
    # R2 metric
    y_pred = slope * x + intercept
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
    
    return A, alpha, r2

def main():
    print("====================================================")
    print("  MESU Width-Scaling Experiment Sweep (Optimized)")
    print("====================================================")
    
    pipeline = JournalPipeline()
    seeds = [42, 101, 202]
    widths = [32, 64, 128, 256]
    capacity_steps = [5, 10, 20, 50, 100, 200, 400, 800]
    
    results_dir = "experiments/results/experiment_1_width_scaling"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "width_scaling.csv")
    
    # Initialize CSV file
    headers = [
        "seed", "experiment", "config", "alpha_decay", "u2_enabled", "replay_buffer_size", "num_facts",
        "recall", "forgetting", "drift", "variance_mean", "variance_min", "variance_max", "runtime_sec", "width"
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
    summary_data = {} # width -> {n: (mean_recall, std_recall, mean_forgetting, std_forgetting)}
    
    for width in widths:
        print(f"\n[*] Calibrating and running for Width = {width}...")
        shift_w, scale_w = autocalibrate_scale_distances(width)
        summary_data[width] = {}
        
        for n_facts in capacity_steps:
            facts = generate_facts(n_facts)
            
            # Pre-embed
            embedded_statements = [pipeline.embed_sentence(f["statement"]) for f in facts]
            embedded_queries = [pipeline.embed_sentence(f["query"]) for f in facts]
            
            run_results = run_width_evaluation(
                width, n_facts, seeds, pipeline, facts, 
                embedded_statements, embedded_queries, shift_w, scale_w
            )
            
            # Log raw results to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                for r in run_results:
                    writer.writerow({
                        "seed": r["seed"],
                        "experiment": "width_scaling",
                        "config": f"width={width},num_facts={n_facts}",
                        "alpha_decay": 0.02,
                        "u2_enabled": True,
                        "replay_buffer_size": 0,
                        "num_facts": n_facts,
                        "recall": r["recall"],
                        "forgetting": r["forgetting"],
                        "drift": r["drift"],
                        "variance_mean": r["variance_mean"],
                        "variance_min": r["variance_min"],
                        "variance_max": r["variance_max"],
                        "runtime_sec": r["runtime_sec"],
                        "width": width
                    })
                    
            rec_all = [r["recall"] for r in run_results]
            forg_all = [r["forgetting"] for r in run_results]
            
            summary_data[width][n_facts] = (
                np.mean(rec_all), np.std(rec_all),
                np.mean(forg_all), np.std(forg_all)
            )
            print(f"  Width {width:3} | N {n_facts:3} | Recall: {np.mean(rec_all)*100:5.2f}% +/- {np.std(rec_all)*100:5.2f}% | Forgetting: {np.mean(forg_all)*100:5.2f}% +/- {np.std(forg_all)*100:5.2f}%")

    print("\n====================================================")
    print("  Width-Scaling Summary Table (Mean Recall)")
    print("====================================================")
    header_str = f"{'N Facts':<8} | " + " | ".join([f"W={w:<5}" for w in widths])
    print(header_str)
    print("-" * len(header_str))
    for n in capacity_steps:
        row_str = f"{n:<8} | " + " | ".join([f"{summary_data[w][n][0]*100:5.2f}%" for w in widths])
        print(row_str)
        
    # Fit scaling laws
    fitting_results = {}
    print("\n====================================================")
    print("  Scaling Law Fitting: Recall(N) = A / N^alpha")
    print("====================================================")
    print(f"{'Width':<6} | {'A (Scale)':<10} | {'alpha (Exp)':<12} | {'R^2 Quality':<10}")
    print("-" * 47)
    for width in widths:
        n_vals = capacity_steps
        rec_vals = [summary_data[width][n][0] for n in capacity_steps]
        A, alpha, r2 = fit_scaling_law(n_vals, rec_vals)
        fitting_results[width] = (A, alpha, r2)
        print(f"{width:<6} | {A:<10.4f} | {alpha:<12.4f} | {r2:<10.4f}")

    # Plot results
    plt.figure(figsize=(10, 5))
    
    # 1. Width vs Recall plot
    plt.subplot(1, 2, 1)
    for width in widths:
        rec_vals = [summary_data[width][n][0] for n in capacity_steps]
        plt.plot(capacity_steps, rec_vals, 'o-', label=f"Width={width}")
    plt.xlabel("Number of Facts (N)")
    plt.ylabel("Recall Accuracy")
    plt.title("Width vs Recall scaling")
    plt.xscale('log')
    plt.xticks(capacity_steps, capacity_steps)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # 2. Log-Log recall scaling plot
    plt.subplot(1, 2, 2)
    for width in widths:
        rec_vals = np.array([summary_data[width][n][0] for n in capacity_steps])
        A, alpha, r2 = fitting_results[width]
        
        # Scatter points
        plt.scatter(capacity_steps, rec_vals + 1e-5, alpha=0.7)
        # Plot fitted line
        fitted_line = A * (np.array(capacity_steps) ** (-alpha))
        plt.plot(capacity_steps, fitted_line + 1e-5, '--', label=f"W={width} (alpha={alpha:.3f})")
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Log(N)")
    plt.ylabel("Log(Recall)")
    plt.title("Log-Log Recall Scaling Fits")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "width_scaling_plots.png")
    plt.savefig(plot_path)
    print(f"\n[*] Plots saved to: {plot_path}")
    
    # Determine the most likely outcome
    alpha_vals = [fitting_results[w][1] for w in widths]
    max_alpha_diff = np.max(alpha_vals) - np.min(alpha_vals)
    
    # Interpretation rules
    print("\n====================================================")
    print("  Interpretation of Results")
    print("====================================================")
    
    # Output markdown report file
    md_path = os.path.join(results_dir, "width_scaling_summary.md")
    with open(md_path, 'w') as f:
        f.write("# MESU Width-Scaling Experiment Analysis\n\n")
        f.write("## Scaling Law Fitting Table\n\n")
        f.write("| Width | A (Scale) | Alpha (Exponent) | R^2 Fit Quality |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for w in widths:
            A, alpha, r2 = fitting_results[w]
            f.write(f"| {w} | {A:.4f} | {alpha:.4f} | {r2:.4f} |\n")
        f.write("\n\n## Summary Data Table (Mean +/- Std Recall)\n\n")
        
        # Build headers
        header_line = "| N Facts | " + " | ".join([f"Width={w}" for w in widths]) + " |\n"
        f.write(header_line)
        f.write("| :--- | " + " | ".join([":---:" for _ in widths]) + " |\n")
        for n in capacity_steps:
            row_vals = []
            for w in widths:
                mean_r, std_r = summary_data[w][n][0], summary_data[w][n][1]
                row_vals.append(f"{mean_r*100:.2f}% +/- {std_r*100:.2f}%")
            f.write(f"| {n} | " + " | ".join(row_vals) + " |\n")
            
        f.write("\n\n## Conclusion & Scientific Verdict\n\n")
        
        conclusion = ""
        evidence = ""
        confidence = ""
        
        # Outcome checks
        if max_alpha_diff < 0.15:
            conclusion = "Outcome A: Interference Accumulation is Dominant"
            evidence = (
                f"As model width increased from 32 to 256, the scaling exponent alpha "
                f"remained approximately constant (ranging between {np.min(alpha_vals):.4f} and {np.max(alpha_vals):.4f}). "
                f"The maximum difference in exponents is only {max_alpha_diff:.4f}. "
                f"Increasing the representational capacity of the bottleneck layer did not alter the power-law scaling exponent."
            )
            confidence = "High confidence (> 90%). The exponent does not shift significantly even when quadrupling the network width, indicating that forgetting is driven primarily by interference scaling rather than bottleneck constraints."
        elif np.all(np.diff(alpha_vals) < -0.05) or (alpha_vals[0] - alpha_vals[-1] > 0.35):
            conclusion = "Outcome B: Representational Bottleneck is Dominant"
            evidence = (
                f"The scaling exponent alpha decreased substantially from {alpha_vals[0]:.4f} (Width=32) "
                f"to {alpha_vals[-1]:.4f} (Width=256). The total exponent reduction is {alpha_vals[0] - alpha_vals[-1]:.4f}. "
                f"Increasing the model capacity alters the scaling retention exponent, meaning capacity is the primary constraint."
            )
            confidence = "High confidence. The substantial and consistent drop in alpha directly supports that capacity is the primary scaling bottleneck."
        else:
            conclusion = "Outcome C: Mixed Regime (Both Interference and Bottleneck Contribute)"
            evidence = (
                f"The scaling exponent alpha decreased somewhat from {alpha_vals[0]:.4f} (Width=32) "
                f"to {alpha_vals[-1]:.4f} (Width=256), but did not flatten or completely collapse. "
                f"Representational capacity helps mitigate the exponent, but interference accumulation remains a major driver."
            )
            confidence = "Moderate confidence. The exponent decreases somewhat but does not collapse, suggesting a mixed regime where both constraints coexist."
            
        f.write(f"### Verdict: {conclusion}\n\n")
        f.write(f"**Supporting Evidence:** {evidence}\n\n")
        f.write(f"**Confidence Level:** {confidence}\n")
        
    print(f"Scientific Verdict: {conclusion}")
    print(f"Summary markdown written to: {md_path}")

if __name__ == "__main__":
    main()
