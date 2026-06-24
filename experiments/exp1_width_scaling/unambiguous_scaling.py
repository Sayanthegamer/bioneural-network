"""
Experiment 5: Unambiguous Schema Scaling & True Capacity Validation
==================================================================
Evaluates the representational capacity of prototypical memory using an
alias-free schema up to N=3200, sweeping Random and Oracle-SVD projections.
"""

import os
import sys
import csv
import time
import argparse
import torch
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# ==============================================================================
# 1. REPRODUCIBILITY
# ==============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 2. ALIAS-FREE FACT GENERATION
# ==============================================================================

def generate_unambiguous_facts(N):
    """
    Generates N alias-free facts using large pools of size 100 to ensure
    unique combinations with max_alias == 1 and duplicate_queries == 0.
    Uses natural language combinations to avoid token collision in SentenceTransformer.
    """
    room_adjs = ["secret", "secure", "hidden", "private", "public", "internal", "external", "upper", "lower", "central"]
    room_nouns = ["vault", "lobby", "lab", "archive", "server", "office", "warehouse", "depot", "hangar", "observatory"]
    rooms = [f"{adj} {noun}" for adj in room_adjs for noun in room_nouns] # 100

    entrance_adjs = ["primary", "secondary", "emergency", "service", "loading", "security", "ventilation", "roof", "basement", "elevator"]
    entrance_nouns = ["door", "entrance", "exit", "gate", "hatch", "shaft", "dock", "tunnel", "passageway", "portal"]
    entrances = [f"{adj} {noun}" for adj in entrance_adjs for noun in entrance_nouns] # 100

    color_adjs = ["light", "dark", "pale", "bright", "deep", "muted", "vibrant", "soft", "neon", "pastel"]
    color_nouns = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "gray", "pink", "teal"]
    colors = [f"{adj} {noun}" for adj in color_adjs for noun in color_nouns] # 100

    project_prefixes = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]
    project_suffixes = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]
    projects = [f"Project {p} {s}" for p in project_prefixes for s in project_suffixes] # 100

    loc_adjs = ["cabinet", "drawer", "shelf", "safe", "bin", "tray", "locker", "box", "chest", "compartment"]
    loc_indices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    locations = [f"{adj} {idx}" for adj in loc_adjs for idx in loc_indices] # 100

    coordinators = [
        "Sarah", "David", "Emma", "James", "Sophia", "Daniel", "Olivia", "Michael", "Isabella", "William",
        "John", "Robert", "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica",
        "Joseph", "Thomas", "Charles", "Christopher", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul",
        "Andrew", "Joshua", "Kenneth", "Kevin", "Brian", "George", "Edward", "Ronald", "Timothy", "Jason",
        "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin",
        "Karen", "Nancy", "Lisa", "Betty", "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily",
        "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura",
        "Cynthia", "Kathleen", "Amy", "Shirley", "Angela", "Helen", "Anna", "Brenda", "Pamela", "Nicole",
        "Samantha", "Katherine", "Christine", "Debra", "Rachel", "Carolyn", "Janet", "Catherine", "Maria", "Heather",
        "Diane", "Ruth", "Julie", "Olive", "Jack", "Jacky", "Harry", "Albert", "Arthur", "Walter", "Fred"
    ][:100] # 100

    sess_adjs = ["daily", "weekly", "monthly", "quarterly", "annual", "interactive", "collaborative", "urgent", "informal", "formal"]
    sess_types = ["sync", "review", "planning", "alignment", "retrospective", "briefing", "workshop", "interview", "debrief", "consultation"]
    sessions = [f"{adj} {stype}" for adj in sess_adjs for stype in sess_types] # 100

    days = [f"day {i}" for i in range(100)]
    hours = [f"hour {i}" for i in range(100)]

    facts = []
    random_gen = random.Random(12345)

    for i in range(N):
        num = random_gen.randint(1000, 9999)
        template_type = i % 3

        if template_type == 0:
            room = rooms[i % 100]
            entrance = entrances[(i // 100) % 100]
            statements = [
                f"The access code for the {room} {entrance} is {num}.",
                f"Entering {room} {entrance} requires keying in {num}.",
                f"To unlock the door to {room} {entrance}, type in {num}.",
                f"The digit sequence {num} grants entry to {room} {entrance}.",
                f"Passcode {num} is configured for access to {room} {entrance}.",
                f"You must enter keycode {num} to access {room} {entrance}.",
                f"The entry pin for the {room} {entrance} is registered as {num}.",
                f"Use password {num} when entering the {room} {entrance}.",
                f"To get into {room} {entrance}, type the code {num} on the keypad.",
                f"The lock on {room} {entrance} opens with passcode {num}."
            ]
            query = f"Which combination of numbers unlocks the {entrance} of the {room}?"

        elif template_type == 1:
            color = colors[i % 100]
            project = projects[(i // 100) % 100]
            loc = locations[(i // 10000) % 100]
            statements = [
                f"The {color} folder for Project {project} is in {loc}.",
                f"You can find the Project {project} paperwork of color {color} inside {loc}.",
                f"The {loc} contains the {color}-colored documents for Project {project}.",
                f"I stored the Project {project} file colored {color} in {loc}.",
                f"The {color} Project {project} folder is resting inside {loc}.",
                f"Go check {loc} for the {color} folder belonging to Project {project}.",
                f"The {color} file for Project {project} is kept in {loc}.",
                f"Project {project} has its {color} dossier placed in {loc}.",
                f"Inside {loc}, you will locate the {color} folder for Project {project}.",
                f"The {color} folder representing Project {project} was left in {loc}."
            ]
            query = f"Where should I look for the documents associated with the {color} Project {project}?"

        else:
            coord = coordinators[i % 100]
            session = sessions[(i // 100) % 100]
            day = days[(i // 10000) % 100]
            hour = hours[(i // 10000) % 100]
            statements = [
                f"{coord}'s {session} meeting is {day} at {hour}.",
                f"{coord} holds the {session} session on {day} scheduled for {hour}.",
                f"The schedule lists {coord}'s {session} on {day} at {hour}.",
                f"We meet with {coord} for the {session} on {day} at {hour}.",
                f"At {hour} on {day}, {coord} conducts the {session} meeting.",
                f"On {day} at {hour}, the {session} session with {coord} begins.",
                f"{coord} runs the {session} gathering on {day} starting at {hour}.",
                f"{coord}'s schedule includes the {session} on {day} at {hour}.",
                f"{coord} will meet us for {session} on {day} at {hour}.",
                f"The {session} with {coord} takes place on {day} at {hour}."
            ]
            query = f"What is the day and hour of the {session} session hosted by {coord}?"

        facts.append({
            "label": i,
            "statements": statements,
            "query": query,
            "template_type": template_type
        })
    return facts

# ==============================================================================
# 3. EVALUATION FUNCTION
# ==============================================================================

def evaluate_projection(statements_emb, queries_emb, facts, projection_type, W=None, statements_mean=None, V_W=None, R=None):
    """
    Evaluates retrieval performance and metrics under the specified projection type.
    """
    centroids_384 = statements_emb.mean(dim=1)
    
    if projection_type == "raw":
        centroids = centroids_384
        queries = queries_emb
    elif projection_type == "random":
        centroids = centroids_384 @ R.T
        queries = queries_emb @ R.T
    elif projection_type == "oracle_svd":
        centroids = (centroids_384 - statements_mean) @ V_W.T
        queries = (queries_emb - statements_mean) @ V_W.T
    
    # Compute L1 distance between queries and centroids
    dists = torch.cdist(queries.unsqueeze(0), centroids.unsqueeze(0), p=1).squeeze(0) # [N, N]
    
    N = len(facts)
    sorted_indices = dists.argsort(dim=1)
    
    ranks = []
    margins = []
    gap_ratios = []
    prototype_densities = []
    decision_gaps = []
    
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    
    for q_idx in range(N):
        correct_label = facts[q_idx]["label"]
        sorted_ind = sorted_indices[q_idx].tolist()
        
        rank = sorted_ind.index(correct_label) + 1
        ranks.append(rank)
        
        if rank == 1:
            recall_at_1 += 1
        if rank <= 5:
            recall_at_5 += 1
        if rank <= 10:
            recall_at_10 += 1
            
        same_class_dist = dists[q_idx, correct_label].item()
        
        # nearest other distance
        mask = torch.ones(N, dtype=torch.bool)
        mask[correct_label] = False
        nearest_other_dist = dists[q_idx, mask].min().item()
        
        margin = nearest_other_dist - same_class_dist
        margins.append(margin)
        
        gap_ratio = nearest_other_dist / (same_class_dist + 1e-8)
        gap_ratios.append(gap_ratio)
        
        # prototype density: mean L1 distance to 10 nearest prototypes
        sorted_dists = dists[q_idx].sort()[0]
        density = sorted_dists[:min(10, N)].mean().item()
        prototype_densities.append(density)

        # decision gap: top2_dist - top1_dist
        top1_dist = sorted_dists[0].item()
        top2_dist = sorted_dists[1].item() if N > 1 else top1_dist
        decision_gaps.append(top2_dist - top1_dist)
        
    return {
        "recall_at_1": recall_at_1 / N,
        "recall_at_5": recall_at_5 / N,
        "recall_at_10": recall_at_10 / N,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_margin": float(np.mean(margins)),
        "p5_margin": float(np.percentile(margins, 5)),
        "p10_margin": float(np.percentile(margins, 10)),
        "mean_gap_ratio": float(np.mean(gap_ratios)),
        "p5_gap_ratio": float(np.percentile(gap_ratios, 5)),
        "p10_gap_ratio": float(np.percentile(gap_ratios, 10)),
        "mean_prototype_density": float(np.mean(prototype_densities)),
        "p5_prototype_density": float(np.percentile(prototype_densities, 5)),
        "p10_prototype_density": float(np.percentile(prototype_densities, 10)),
        "mean_decision_gap": float(np.mean(decision_gaps)),
        "p5_decision_gap": float(np.percentile(decision_gaps, 5)),
        "p10_decision_gap": float(np.percentile(decision_gaps, 10))
    }

# ==============================================================================
# 4. REPORT & PLOTTING
# ==============================================================================

def generate_report(results_by_N, md_path, capacity_breakpoints):
    """Generates the Markdown report with SVD leakage disclosures."""
    lines = []
    lines.append("# Experiment 5: Unambiguous Schema Scaling & True Capacity Validation\n")
    lines.append("## Capacity Breakdown Points ($N^*$)")
    lines.append("Defined as the largest $N \\in \\{100, 200, 400, 800, 1600, 3200\\}$ where $\\text{Recall@1} \\ge 95\\%$ and $\\text{p5 Margin} \\ge 0$.\n")
    
    lines.append("| Projection Configuration | $N^*$ |")
    lines.append("| :--- | :---: |")
    for key, val in capacity_breakpoints.items():
        lines.append(f"| {key} | {val} |")
    lines.append("\n")

    lines.append("> [!IMPORTANT]")
    lines.append("> **Oracle-SVD Data Leakage Disclaimer:** The Oracle-SVD projection matrix is computed based on the same statement corpus being retrieved.")
    lines.append("> Therefore, Oracle-SVD results serve as an upper-bound representational ceiling rather than a realistic deployment scenario.\n")

    lines.append("## Detailed Sweep Results\n")
    lines.append("| N | Projection | Width (W) | Recall@1 | Recall@10 | Mean Rank | p5 Margin | Mean Gap Ratio | Mean Decision Gap | Mean Density |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for N in sorted(results_by_N.keys()):
        for config in results_by_N[N]:
            r = config["results"]
            w_str = str(config["W"]) if config["W"] is not None else "-"
            
            # Print recall as Mean +- Std if std > 0
            if "std_recall_at_1" in r and r["std_recall_at_1"] > 0:
                recall_str = f"{r['recall_at_1']*100:.1f}% ± {r['std_recall_at_1']*100:.1f}%"
            else:
                recall_str = f"{r['recall_at_1']*100:.1f}%"
                
            lines.append(f"| {N} | {config['type']} | {w_str} | {recall_str} | "
                         f"{r['recall_at_10']*100:.1f}% | {r['mean_rank']:.1f} | {r['p5_margin']:.2f} | "
                         f"{r['mean_gap_ratio']:.2f} | {r['mean_decision_gap']:.2f} | {r['mean_prototype_density']:.2f} |")
            
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  [+] Markdown report written to {md_path}")

def generate_plots(results_by_N, plot_path):
    """Generates scaling plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Experiment 5: Scaling Metrics under Unambiguous Schema", fontsize=14, fontweight='bold')
    
    N_vals = sorted(results_by_N.keys())
    
    # Define plotting series
    series_configs = [
        ("raw", None, "Raw (384D)", "black", "o"),
        ("random", 32, "Random (W=32)", "red", "s"),
        ("random", 64, "Random (W=64)", "orange", "^"),
        ("random", 128, "Random (W=128)", "yellow", "v"),
        ("oracle_svd", 32, "Oracle-SVD (W=32)", "blue", "D"),
        ("oracle_svd", 64, "Oracle-SVD (W=64)", "teal", "p"),
        ("oracle_svd", 128, "Oracle-SVD (W=128)", "green", "*"),
    ]
    
    # Plot 1: Recall@1 vs N
    ax = axes[0]
    for p_type, W, label, color, marker in series_configs:
        vals = []
        for N in N_vals:
            for cfg in results_by_N[N]:
                if cfg["type"] == p_type and cfg["W"] == W:
                    vals.append(cfg["results"]["recall_at_1"] * 100)
        ax.plot(N_vals, vals, marker + '-', label=label, color=color)
    ax.set_xscale('log')
    ax.set_xticks(N_vals)
    ax.set_xticklabels([str(n) for n in N_vals])
    ax.set_xlabel("Number of Facts (N)")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title("Recall@1 vs N")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Plot 2: p5 Margin vs N
    ax = axes[1]
    for p_type, W, label, color, marker in series_configs:
        vals = []
        for N in N_vals:
            for cfg in results_by_N[N]:
                if cfg["type"] == p_type and cfg["W"] == W:
                    vals.append(cfg["results"]["p5_margin"])
        ax.plot(N_vals, vals, marker + '-', label=label, color=color)
    ax.set_xscale('log')
    ax.set_xticks(N_vals)
    ax.set_xticklabels([str(n) for n in N_vals])
    ax.set_xlabel("Number of Facts (N)")
    ax.set_ylabel("p5 Margin")
    ax.set_title("p5 Margin vs N")
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean Decision Gap vs N
    ax = axes[2]
    for p_type, W, label, color, marker in series_configs:
        vals = []
        for N in N_vals:
            for cfg in results_by_N[N]:
                if cfg["type"] == p_type and cfg["W"] == W:
                    vals.append(cfg["results"]["mean_decision_gap"])
        ax.plot(N_vals, vals, marker + '-', label=label, color=color)
    ax.set_xscale('log')
    ax.set_xticks(N_vals)
    ax.set_xticklabels([str(n) for n in N_vals])
    ax.set_xlabel("Number of Facts (N)")
    ax.set_ylabel("Mean Decision Gap")
    ax.set_title("Mean Decision Gap vs N")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [+] Scaling plots saved to {plot_path}")

# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Unambiguous Schema Scaling")
    parser.add_argument("--N_max", type=int, default=3200, help="Maximum N to evaluate")
    args, _ = parser.parse_known_args()
    
    set_seed(42)
    
    print("====================================================")
    print("  Experiment 5: Unambiguous Schema Scaling Sweeps")
    print("====================================================")
    
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_5_unambiguous_scaling")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "unambiguous_scaling.csv")
    plot_path = os.path.join(results_dir, "unambiguous_scaling.png")
    md_path = os.path.join(results_dir, "unambiguous_scaling.md")
    
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    print("[*] Loading SentenceTransformer backbone...")
    bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    bi_encoder.to('cpu')
    
    N_values = [n for n in [100, 200, 400, 800, 1600, 3200] if n <= args.N_max]
    widths = [32, 64, 128]
    
    results_by_N = {}
    
    for N in N_values:
        print(f"\nRunning N = {N} ...")
        facts = generate_unambiguous_facts(N)
        
        # Verify alias-free properties
        query_to_labels = defaultdict(list)
        for f in facts:
            query_to_labels[f["query"]].append(f["label"])
        max_alias = max(len(labels) for labels in query_to_labels.values())
        duplicate_queries = sum(1 for q, labels in query_to_labels.items() if len(labels) > 1)
        assert max_alias == 1, f"Assertion failed: max_alias is {max_alias} (expected 1)"
        assert duplicate_queries == 0, f"Assertion failed: duplicate_queries is {duplicate_queries} (expected 0)"
        
        # Encode statement and queries
        all_statement_texts = [s for f in facts for s in f["statements"]]
        all_query_texts = [f["query"] for f in facts]
        
        with torch.no_grad():
            statements_flat = bi_encoder.encode(all_statement_texts, batch_size=256, convert_to_tensor=True, device='cpu')
            statements_emb = statements_flat.view(N, 10, -1)
            queries_emb = bi_encoder.encode(all_query_texts, batch_size=256, convert_to_tensor=True, device='cpu')
            
        # Fit SVD projection matrix V_W for largest width (128)
        statements_mean = statements_flat.mean(dim=0, keepdim=True)
        statements_centered = statements_flat - statements_mean
        _, _, Vh = torch.linalg.svd(statements_centered, full_matrices=False)
        
        # Evaluate configurations
        configs_evaluated = []
        
        # 1. Raw
        print("  [*] Evaluating Raw 384D...")
        raw_res = evaluate_projection(statements_emb, queries_emb, facts, "raw")
        raw_res["std_recall_at_1"] = 0.0
        configs_evaluated.append({"type": "raw", "W": None, "results": raw_res})
        
        for W in widths:
            # 2. Random projection (evaluated across seeds 42, 101, 202)
            print(f"  [*] Evaluating Random (W={W}) across 3 seeds...")
            rand_res_list = []
            for seed in [42, 101, 202]:
                gen_state = torch.random.get_rng_state()
                torch.manual_seed(seed + W)
                R = torch.randn(W, 384)
                R = R / torch.linalg.norm(R, dim=1, keepdim=True)
                torch.random.set_rng_state(gen_state)
                
                res = evaluate_projection(statements_emb, queries_emb, facts, "random", W=W, R=R)
                rand_res_list.append(res)
                
            rand_res = {}
            for key in rand_res_list[0].keys():
                vals = [r[key] for r in rand_res_list]
                rand_res[key] = float(np.mean(vals))
            rand_res["std_recall_at_1"] = float(np.std([r["recall_at_1"] for r in rand_res_list]))
            configs_evaluated.append({"type": "random", "W": W, "results": rand_res})
            
            # 3. Oracle-SVD projection
            print(f"  [*] Evaluating Oracle-SVD (W={W})...")
            V_W = Vh[:W]
            svd_res = evaluate_projection(statements_emb, queries_emb, facts, "oracle_svd", W=W, statements_mean=statements_mean, V_W=V_W)
            svd_res["std_recall_at_1"] = 0.0
            configs_evaluated.append({"type": "oracle_svd", "W": W, "results": svd_res})
            
        results_by_N[N] = configs_evaluated
        
        # Append to CSV
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, 'a', newline='') as f:
            headers = ["N", "projection_type", "W"] + list(raw_res.keys())
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            for config in configs_evaluated:
                r = config["results"]
                row = [N, config["type"], config["W"]] + [r[k] for k in headers[3:]]
                writer.writerow(row)
                
    # Calculate Capacity Breakdown Point N* for each type
    capacity_breakpoints = {}
    
    keys = ["Raw (384D)", "Random (W=32)", "Random (W=64)", "Random (W=128)", "Oracle-SVD (W=32)", "Oracle-SVD (W=64)", "Oracle-SVD (W=128)"]
    for key in keys:
        if "Raw" in key:
            ptype, W = "raw", None
        else:
            ptype = "random" if "Random" in key else "oracle_svd"
            W = int(key.split("=")[1].replace(")", ""))
            
        n_star = "< 100"
        for N in sorted(N_values):
            res = None
            for cfg in results_by_N[N]:
                if cfg["type"] == ptype and cfg["W"] == W:
                    res = cfg["results"]
                    break
            if res and res["recall_at_1"] >= 0.95 and res["p5_margin"] >= 0:
                n_star = str(N)
        capacity_breakpoints[key] = n_star

    print("\n====================================================")
    print("  Capacity Breakdown Points (N*) Summary")
    print("====================================================")
    for key, val in capacity_breakpoints.items():
        print(f"  {key:15s} : N* = {val}")
        
    generate_report(results_by_N, md_path, capacity_breakpoints)
    generate_plots(results_by_N, plot_path)
    
    print("\nComplete!")
    print(f"CSV:    {csv_path}")
    print(f"Plots:  {plot_path}")
    print(f"Report: {md_path}")

if __name__ == "__main__":
    main()
