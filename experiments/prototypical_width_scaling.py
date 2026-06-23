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
    shift = -mean_val
    scale = 2.0 * std_val if std_val > 1e-5 else 1.0
    return shift, scale

# Autocalibrate globally for input dim 384
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
            "label": i,
            "template_type": template_type
        })
    return facts

def classify_knn(query_z, exemplar_tensors, labels, k=1):
    diff = exemplar_tensors - query_z.unsqueeze(0)
    dists = diff.abs().sum(dim=1)
    topk_indices = dists.argsort()[:k]
    topk_labels = [labels[idx] for idx in topk_indices.tolist()]
    return max(set(topk_labels), key=topk_labels.count)

def fit_scaling_law(n_vals, recall_vals):
    x = np.log(np.array(n_vals))
    eps = 1e-5
    y = np.log(np.array(recall_vals) + eps)
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -slope
    A = np.exp(intercept)
    
    y_pred = slope * x + intercept
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
    return A, alpha, r2

def estimate_n50(capacity_steps, mean_recalls, mean_margins):
    n50_recall = None
    for i in range(len(capacity_steps) - 1):
        n_prev, n_next = capacity_steps[i], capacity_steps[i+1]
        r_prev, r_next = mean_recalls[i], mean_recalls[i+1]
        if r_prev >= 0.5 and r_next < 0.5:
            t = (0.5 - r_prev) / (r_next - r_prev + 1e-8)
            n50_recall = n_prev + t * (n_next - n_prev)
            break
    if n50_recall is None:
        if mean_recalls[-1] >= 0.5:
            n50_recall = float(capacity_steps[-1])
        else:
            n50_recall = float(capacity_steps[0])
            
    n0_margin = None
    for i in range(len(capacity_steps) - 1):
        n_prev, n_next = capacity_steps[i], capacity_steps[i+1]
        m_prev, m_next = mean_margins[i], mean_margins[i+1]
        if m_prev >= 0.0 and m_next < 0.0:
            t = (0.0 - m_prev) / (m_next - m_prev + 1e-8)
            n0_margin = n_prev + t * (n_next - n_prev)
            break
    if n0_margin is None:
        if mean_margins[-1] >= 0.0:
            n0_margin = float(capacity_steps[-1])
        else:
            n0_margin = float(capacity_steps[0])
            
    return min(n50_recall, n0_margin)

def main():
    print("====================================================")
    print("  Prototypical Bottleneck Width-Scaling Sweep")
    print("====================================================")
    
    pipeline = JournalPipeline()
    seeds = [42, 101, 202]
    widths = [32, 64, 128, 256]
    capacity_steps = [5, 10, 20, 50, 100, 200, 400, 800]
    
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "prototypical_width_scaling.csv")
    
    headers = [
        "seed", "width", "num_facts", "prototype_acc", "knn1_acc",
        "mean_radius", "mean_nearest_other", "separation_ratio",
        "mean_margin", "median_margin", "p5_margin", "min_margin", "runtime_sec"
    ]
    
    summary_data = {} # (width, n) -> {metrics}
    
    # Check if CSV already exists and has complete data
    use_existing_csv = False
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) >= 97:  # 1 header + 4 widths * 8 steps * 3 seeds = 97 lines
            print("[*] Detected complete existing CSV results. Loading data to compile report and plots...")
            use_existing_csv = True

    if use_existing_csv:
        # Load summary_data from existing CSV
        raw_rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows.append({
                    "width": int(row["width"]),
                    "num_facts": int(row["num_facts"]),
                    "prototype_acc": float(row["prototype_acc"]),
                    "knn1_acc": float(row["knn1_acc"]),
                    "mean_radius": float(row["mean_radius"]),
                    "mean_nearest_other": float(row["mean_nearest_other"]),
                    "separation_ratio": float(row["separation_ratio"]),
                    "mean_margin": float(row["mean_margin"]),
                    "p5_margin": float(row["p5_margin"]),
                    "min_margin": float(row["min_margin"])
                })
        
        # Group by (width, n_facts)
        for width in widths:
            for n_facts in capacity_steps:
                matching_rows = [r for r in raw_rows if r["width"] == width and r["num_facts"] == n_facts]
                if len(matching_rows) > 0:
                    avg_proto = np.mean([r["prototype_acc"] for r in matching_rows])
                    std_proto = np.std([r["prototype_acc"] for r in matching_rows])
                    avg_knn = np.mean([r["knn1_acc"] for r in matching_rows])
                    avg_margin = np.mean([r["mean_margin"] for r in matching_rows])
                    avg_p5_margin = np.mean([r["p5_margin"] for r in matching_rows])
                    avg_min_margin = np.mean([r["min_margin"] for r in matching_rows])
                    avg_sep_ratio = np.mean([r["separation_ratio"] for r in matching_rows])
                    avg_radius = np.mean([r["mean_radius"] for r in matching_rows])
                    avg_nearest = np.mean([r["mean_nearest_other"] for r in matching_rows])
                    
                    summary_data[(width, n_facts)] = {
                        "prototype_acc": avg_proto,
                        "prototype_acc_std": std_proto,
                        "knn1_acc": avg_knn,
                        "mean_margin": avg_margin,
                        "p5_margin": avg_p5_margin,
                        "min_margin": avg_min_margin,
                        "separation_ratio": avg_sep_ratio,
                        "mean_radius": avg_radius,
                        "mean_nearest_other": avg_nearest
                    }
    else:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
        for n_facts in capacity_steps:
            print(f"\n[*] Preparing N = {n_facts} facts...")
            facts = generate_facts(n_facts)
            
            # Pre-embed facts to reuse across widths and seeds (optimization)
            embedded_statements = [pipeline.embed_sentence(f["statement"]) for f in facts]
            embedded_queries = [pipeline.embed_sentence(f["query"]) for f in facts]
            
            for width in widths:
                print(f"  [-] Evaluating Bottleneck Width = {width}...")
                
                run_results = []
                for seed in seeds:
                    t_start = time.time()
                    set_seed(seed)
                    
                    # Setup frozen backbone representation
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
                    
                    # Ingestion
                    statement_projections = []
                    fact_labels = [f["label"] for f in facts]
                    
                    with torch.no_grad():
                        for idx in range(n_facts):
                            statement_vec = embedded_statements[idx]
                            label = facts[idx]["label"]
                            model.add_fact(statement_vec, label)
                            
                            z_proj = model.backbone(statement_vec).squeeze(0)
                            statement_projections.append(z_proj)
                            
                        statement_projections = torch.stack(statement_projections)
                        query_projections = torch.stack([model.backbone(q_vec).squeeze(0) for q_vec in embedded_queries])
                        
                        # Prototype recall
                        proto_correct = 0
                        for idx in range(n_facts):
                            logits = model(embedded_queries[idx])
                            pred = logits.argmax(dim=-1).item()
                            if pred == facts[idx]["label"]:
                                proto_correct += 1
                        proto_acc = proto_correct / float(n_facts)
                        
                        # 1-NN recall
                        knn1_correct = 0
                        for idx in range(n_facts):
                            pred1 = classify_knn(query_projections[idx], statement_projections, fact_labels, k=1)
                            if pred1 == facts[idx]["label"]:
                                knn1_correct += 1
                        knn1_acc = knn1_correct / float(n_facts)
                        
                        # Geometry metrics
                        radii = []
                        nearest_others = []
                        margins = []
                        for idx in range(n_facts):
                            correct_proto = model.prototypes[idx]
                            q_z = query_projections[idx]
                            radius = (q_z - correct_proto).abs().sum().item()
                            radii.append(radius)
                            
                            other_dists = []
                            for other_idx in range(n_facts):
                                if other_idx != idx:
                                    other_proto = model.prototypes[other_idx]
                                    dist = (q_z - other_proto).abs().sum().item()
                                    other_dists.append(dist)
                            
                            nearest_other = min(other_dists) if len(other_dists) > 0 else 0.0
                            nearest_others.append(nearest_other)
                            margins.append(nearest_other - radius)
                            
                        mean_radius = np.mean(radii)
                        mean_nearest_other = np.mean(nearest_others)
                        sep_ratio = mean_nearest_other / (mean_radius + 1e-8)
                        
                        mean_margin = np.mean(margins)
                        median_margin = np.median(margins)
                        p5_margin = np.percentile(margins, 5)
                        min_margin = np.min(margins)
                        
                        runtime = time.time() - t_start
                        
                        res = {
                            "seed": seed,
                            "width": width,
                            "num_facts": n_facts,
                            "prototype_acc": proto_acc,
                            "knn1_acc": knn1_acc,
                            "mean_radius": mean_radius,
                            "mean_nearest_other": mean_nearest_other,
                            "separation_ratio": sep_ratio,
                            "mean_margin": mean_margin,
                            "median_margin": median_margin,
                            "p5_margin": p5_margin,
                            "min_margin": min_margin,
                            "runtime_sec": runtime
                        }
                        run_results.append(res)
                        
                # Save raw results
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    for r in run_results:
                        writer.writerow(r)
                        
                # Compile stats
                avg_proto = np.mean([r["prototype_acc"] for r in run_results])
                std_proto = np.std([r["prototype_acc"] for r in run_results])
                avg_knn = np.mean([r["knn1_acc"] for r in run_results])
                avg_margin = np.mean([r["mean_margin"] for r in run_results])
                avg_p5_margin = np.mean([r["p5_margin"] for r in run_results])
                avg_min_margin = np.mean([r["min_margin"] for r in run_results])
                avg_sep_ratio = np.mean([r["separation_ratio"] for r in run_results])
                avg_radius = np.mean([r["mean_radius"] for r in run_results])
                avg_nearest = np.mean([r["mean_nearest_other"] for r in run_results])
                
                summary_data[(width, n_facts)] = {
                    "prototype_acc": avg_proto,
                    "prototype_acc_std": std_proto,
                    "knn1_acc": avg_knn,
                    "mean_margin": avg_margin,
                    "p5_margin": avg_p5_margin,
                    "min_margin": avg_min_margin,
                    "separation_ratio": avg_sep_ratio,
                    "mean_radius": avg_radius,
                    "mean_nearest_other": avg_nearest
                }
                
                print(f"    Width {width:3} | Recall: {avg_proto*100:6.2f}% +/- {std_proto*100:5.2f}% | Margin (Mean/P5/Min): {avg_margin:6.4f} / {avg_p5_margin:6.4f} / {avg_min_margin:6.4f} | Sep Ratio: {avg_sep_ratio:.4f}")

    # Compute N50 Capacity Thresholds
    n50_values = []
    print("\n====================================================")
    print("  Effective Capacity Threshold (N50) by Width")
    print("====================================================")
    for width in widths:
        mean_recalls = [summary_data[(width, n)]["prototype_acc"] for n in capacity_steps]
        mean_margins = [summary_data[(width, n)]["mean_margin"] for n in capacity_steps]
        n50 = estimate_n50(capacity_steps, mean_recalls, mean_margins)
        n50_values.append(n50)
        print(f"  Width {width:3} | N50 Threshold = {n50:.2f} facts")
        
    # Fit Capacity Scaling Law: N50 = B * W^beta
    log_widths = np.log(np.array(widths))
    log_n50 = np.log(np.array(n50_values))
    beta, log_B = np.polyfit(log_widths, log_n50, 1)
    B = np.exp(log_B)
    
    # Fit scaling laws Recall(N) = A * N^-alpha for each width
    fitting_results = {}
    for width in widths:
        rec_vals = [summary_data[(width, n)]["prototype_acc"] for n in capacity_steps]
        A, alpha, r2 = fit_scaling_law(capacity_steps, rec_vals)
        fitting_results[width] = (A, alpha, r2)
        
    # Generate Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Recall vs N
    ax = axes[0]
    for width in widths:
        rec_vals = [summary_data[(width, n)]["prototype_acc"] for n in capacity_steps]
        ax.plot(capacity_steps, rec_vals, 'o-', label=f"Width={width} (alpha={fitting_results[width][1]:.3f})")
    ax.axhline(0.5, color='gray', linestyle='--', label='50% Threshold')
    ax.set_xscale('log')
    ax.set_xticks(capacity_steps)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Number of Facts (N)")
    ax.set_ylabel("Recall Accuracy")
    ax.set_title("Prototypical Recall Scaling Curves")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    
    # Plot 2: Mean L1 Margin vs N
    ax = axes[1]
    for width in widths:
        margin_vals = [summary_data[(width, n)]["mean_margin"] for n in capacity_steps]
        ax.plot(capacity_steps, margin_vals, 'o-', label=f"Width={width}")
    ax.axhline(0.0, color='r', linestyle='--', label='0 Margin Crossover')
    ax.set_xscale('log')
    ax.set_xticks(capacity_steps)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Number of Facts (N)")
    ax.set_ylabel("Mean L1 Margin")
    ax.set_title("Mean L1 Margin Crossover Curves")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    
    # Plot 3: Mean L1 Margin vs N/W
    ax = axes[2]
    for width in widths:
        margin_vals = [summary_data[(width, n)]["mean_margin"] for n in capacity_steps]
        ratio_vals = [n / float(width) for n in capacity_steps]
        ax.plot(ratio_vals, margin_vals, 'o-', label=f"Width={width}")
    ax.axhline(0.0, color='r', linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel("Facts per Dimension (N / Width)")
    ax.set_ylabel("Mean L1 Margin")
    ax.set_title("Facts-per-Dimension Normalization Collapse")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "prototypical_width_scaling_plots.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n[*] Plots saved to: {plot_path}")
    
    # Determine Scientific Verdict
    verdict = ""
    evidence = ""
    if beta < 0.2:
        verdict = "Width Saturation / Minimal Benefit (beta ~ 0)"
        evidence = f"The capacity exponent beta is {beta:.4f}, meaning scaling representation bottleneck width does not resolve crowding."
    elif beta < 0.8:
        verdict = "Sublinear Scaling / Diminishing Returns (beta ~ 0.5)"
        evidence = f"The capacity exponent beta is {beta:.4f}, implying capacity scales sublinearly with width (likely due to MiniLM manifold dimensionality constraints)."
    elif beta < 1.2:
        verdict = "Linear Capacity Scaling (beta ~ 1.0)"
        evidence = f"The capacity exponent beta is {beta:.4f}, confirming that double the width approximately yields double the prototype capacity."
    else:
        verdict = "Superlinear Capacity Scaling (beta > 1.2)"
        evidence = f"The capacity exponent beta is {beta:.4f}, showing that scaling representation dimensions yields superlinear capacity improvements."
        
    print(f"Scientific Verdict: {verdict}")
    
    # Output markdown report file
    md_path = os.path.join(results_dir, "prototypical_width_scaling_summary.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Prototypical Bottleneck Width-Scaling Sweep Analysis\n\n")
        
        f.write("## 🏆 The Capacity Scaling Verdict\n\n")
        f.write(f"**Scaling Law Equation:** $N_{{50}} = {B:.4f} \\cdot W^{{{beta:.4f}}}$\n\n")
        f.write(f"### Verdict: {verdict}\n\n")
        f.write(f"**Evidence:** {evidence}\n\n")
        
        f.write("## 📊 Capacity Threshold Summary Table\n\n")
        f.write("| Width (W) | Interpolated N50 Capacity Threshold | Power-Law Recall Exponent (alpha) | R^2 Fit Quality |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for i, w in enumerate(widths):
            f.write(f"| **{w}** | {n50_values[i]:.2f} facts | {fitting_results[w][1]:.4f} | {fitting_results[w][2]:.4f} |\n")
            
        f.write("\n\n## 📝 Comprehensive Telemetry Table\n\n")
        
        f.write("| Width | N Facts | Recall (Mean +/- Std) | Mean Margin | P5 Margin | Min Margin | Sep Ratio | Radius | Nearest Other |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for w in widths:
            for n in capacity_steps:
                d = summary_data[(w, n)]
                f.write(f"| {w} | {n} | {d['prototype_acc']*100:.2f}% +/- {d['prototype_acc_std']*100:.2f}% | {d['mean_margin']:.4f} | {d['p5_margin']:.4f} | {d['min_margin']:.4f} | {d['separation_ratio']:.4f} | {d['mean_radius']:.4f} | {d['mean_nearest_other']:.4f} |\n")
                
        f.write("\n\n## 🔍 Interpretation and Implications\n\n")
        f.write("1. **Is Exponent Shift Present?** We can compare the power-law decay exponents $\\alpha$ across widths. If $\\alpha$ drops as $W$ increases, we have successfully flattened the forgetting curve using only representation scaling.\n")
        f.write("2. **Does N/W Normalization Collapse?** In the third panel of the plots, if curves collapse on top of each other, it proves that crowding is a universal geometric function of *facts-per-dimension* ($N/W$).\n")
        f.write("3. **Percentile Margin early warning:** Keep an eye on `P5 Margin` and `Min Margin`. If they turn negative before the mean margin does, local cluster interference is occurring, preceding global recall degradation.\n")
        
    print(f"Summary report written to: {md_path}")

if __name__ == "__main__":
    main()
