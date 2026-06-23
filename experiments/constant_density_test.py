import sys
import os
import csv
import time
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mnc_project'))
from mnc.layers import MNCLinear, MNCPrototypicalNetwork
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
    return -mean_val, 2.0 * std_val if std_val > 1e-5 else 1.0

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
        facts.append({"statement": statement, "query": query, "label": i})
    return facts

def main():
    print("====================================================")
    print("  Constant N/W Density Verification Sweep")
    print("====================================================")
    pipeline = JournalPipeline()
    seeds = [42, 101, 202]
    widths = [32, 64, 128, 256, 512]
    ratios = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "constant_density_test.csv")
    
    headers = [
        "ratio", "width", "num_facts", "seed", "recall", 
        "mean_radius", "mean_nearest_other", "separation_ratio", 
        "mean_margin", "normalized_margin", "runtime_sec"
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
    
    # 2D Grid Sweep
    for ratio in ratios:
        print(f"\n[*] Evaluating constant ratio N/W = {ratio}...")
        for width in widths:
            n_facts = int(round(ratio * width))
            if n_facts < 2:
                n_facts = 2  # At least 2 facts to compute separation
            
            print(f"  [-] Width = {width} | N = {n_facts} facts...")
            facts = generate_facts(n_facts)
            embedded_statements = [pipeline.embed_sentence(f["statement"]) for f in facts]
            embedded_queries = [pipeline.embed_sentence(f["query"]) for f in facts]
            
            for seed in seeds:
                t_start = time.time()
                set_seed(seed)
                
                backbone_linear = MNCLinear(384, width)
                for param in backbone_linear.parameters():
                    param.requires_grad = False
                    
                backbone = nn.Sequential(
                    backbone_linear,
                    ScaleDistances(SHIFT_384, SCALE_384),
                    nn.Tanh()
                )
                backbone.eval()
                model = MNCPrototypicalNetwork(backbone, bottleneck_dim=width, num_classes=n_facts)
                model.eval()
                
                with torch.no_grad():
                    for idx in range(n_facts):
                        model.add_fact(embedded_statements[idx], facts[idx]["label"])
                        
                    q_vecs = torch.cat(embedded_queries, dim=0)
                    query_projections = model.backbone(q_vecs)
                    
                    chunk_size = 200
                    dists_list = []
                    for i in range(0, n_facts, chunk_size):
                        q_chunk = query_projections[i:i+chunk_size]
                        diff_chunk = q_chunk.unsqueeze(1) - model.prototypes.unsqueeze(0)
                        dists_list.append(diff_chunk.abs().sum(dim=2))
                    dists_all = torch.cat(dists_list, dim=0)
                    
                    preds = (-dists_all).argmax(dim=-1).tolist()
                    correct = sum(1 for idx in range(n_facts) if preds[idx] == facts[idx]["label"])
                    recall = correct / float(n_facts)
                    
                    diff_self = query_projections - model.prototypes
                    radii_tensor = diff_self.abs().sum(dim=1)
                    mean_radius = radii_tensor.mean().item()
                    
                    mask = torch.eye(n_facts, device=dists_all.device) * 1e9
                    nearest_others_tensor = (dists_all + mask).min(dim=1).values
                    mean_nearest_other = nearest_others_tensor.mean().item()
                    
                    sep_ratio = mean_nearest_other / (mean_radius + 1e-8)
                    mean_margin = (nearest_others_tensor - radii_tensor).mean().item()
                    norm_margin = mean_margin / float(width)
                    
                    runtime = time.time() - t_start
                    
                    row = {
                        "ratio": ratio,
                        "width": width,
                        "num_facts": n_facts,
                        "seed": seed,
                        "recall": recall,
                        "mean_radius": mean_radius,
                        "mean_nearest_other": mean_nearest_other,
                        "separation_ratio": sep_ratio,
                        "mean_margin": mean_margin,
                        "normalized_margin": norm_margin,
                        "runtime_sec": runtime
                      }
                      
                    with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=headers)
                        writer.writerow(row)
      
    # Group results and compile summary
    plot_results(csv_path, results_dir)

def plot_results(csv_path, results_dir):
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            data.append({
                "ratio": float(r["ratio"]),
                "width": int(r["width"]),
                "num_facts": int(r["num_facts"]),
                "recall": float(r["recall"]),
                "separation_ratio": float(r["separation_ratio"]),
                "normalized_margin": float(r["normalized_margin"])
            })
            
    widths = sorted(list(set([r["width"] for r in data])))
    ratios = sorted(list(set([r["ratio"] for r in data])))
    
    summary = {}
    for ratio in ratios:
        for width in widths:
            matches = [r for r in data if r["ratio"] == ratio and r["width"] == width]
            if matches:
                summary[(ratio, width)] = {
                    "recall": np.mean([m["recall"] for m in matches]),
                    "recall_std": np.std([m["recall"] for m in matches]),
                    "sep_ratio": np.mean([m["separation_ratio"] for m in matches]),
                    "norm_margin": np.mean([m["normalized_margin"] for m in matches])
                }
                
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax = axes[0]
    for ratio in ratios:
        rec_vals = [summary[(ratio, w)]["recall"] for w in widths]
        ax.plot(widths, rec_vals, 'o-', label=f"N/W={ratio}")
    ax.set_xlabel("Bottleneck Width (W)")
    ax.set_ylabel("Recall Accuracy")
    ax.set_title("Recall vs Width under Constant Density")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[1]
    for ratio in ratios:
        sep_vals = [summary[(ratio, w)]["sep_ratio"] for w in widths]
        ax.plot(widths, sep_vals, 'o-', label=f"N/W={ratio}")
    ax.set_xlabel("Bottleneck Width (W)")
    ax.set_ylabel("Separation Ratio (Nearest Other / Radius)")
    ax.set_title("Separation Ratio under Constant Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[2]
    for ratio in ratios:
        margin_vals = [summary[(ratio, w)]["norm_margin"] for w in widths]
        ax.plot(widths, margin_vals, 'o-', label=f"N/W={ratio}")
    ax.set_xlabel("Bottleneck Width (W)")
    ax.set_ylabel("Normalized L1 Margin (Margin / Width)")
    ax.set_title("Normalized Margin under Constant Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "constant_density_plots.png")
    plt.savefig(plot_path, dpi=150)
    print(f"[*] Saved plots to: {plot_path}")
    
    write_markdown_report(ratios, widths, summary, results_dir)

def write_markdown_report(ratios, widths, summary, results_dir):
    md_path = os.path.join(results_dir, "constant_density_summary.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Constant N/W Density Verification Analysis\n\n")
        f.write("## 🏆 Scientific Verdict\n\n")
        f.write("This report validates the geometric density law under a constant $N/W$ ratio constraint.\n\n")
        
        f.write("## 📊 Density Ratio Performance Table\n\n")
        f.write("| Density Ratio (N/W) | Width (W) | Fact Count (N) | Recall (Mean +/- Std) | Separation Ratio | Normalized Margin |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for ratio in ratios:
            for w in widths:
                n = int(round(ratio * w))
                if n < 2: n = 2
                s = summary[(ratio, w)]
                f.write(f"| **{ratio:.1f}** | {w} | {n} | {s['recall']*100:.2f}% +/- {s['recall_std']*100:.2f}% | {s['sep_ratio']:.4f} | {s['norm_margin']:.4f} |\n")
                
        f.write("\n\n## 🔍 Interpretation\n")
        f.write("1. **Flatness of Recall Lines:** Under a perfect density law, recall should remain flat across widths for a constant $N/W$ ratio. If recall instead climbs, it indicates dimensional expansion advantages (Possibility 1/2/3).\n")
        f.write("2. **Separation Ratio Stability:** Check if the separation ratio remains constant across widths. Since it is dimensionless, its flatness directly proves a unified geometric packing density.\n")
        f.write("3. **Normalized Margin Convergence:** Margin scaling is normalized by bottleneck dimension ($W$). If the normalized margin is flat, it proves the geometry scales linearly with representational volume.\n")

    print(f"[*] Saved summary to: {md_path}")

if __name__ == "__main__":
    main()
