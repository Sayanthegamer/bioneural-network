"""
Kaggle Notebook Version: Experiment 6
================================================================================
Self-contained script for Kaggle running, fully GPU-accelerated (CUDA).
"""

import os
import sys
import csv
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# ==============================================================================
# 1. REPRODUCIBILITY & SETUP
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
# 3. LEARNED PROJECTION MODEL & TRAINING
# ==============================================================================

class LearnedProjection(nn.Module):
    def __init__(self, in_features, out_features, proj_type="linear"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj_type = proj_type
        
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.W, mean=0.0, std=1.0)
        with torch.no_grad():
            self.W.data = self.W.data / (self.W.data.norm(p=2, dim=1, keepdim=True) + 1e-8)

    def forward(self, x):
        if self.proj_type == "linear":
            return x @ self.W.T
        elif self.proj_type == "mnc":
            # MNC-native L1 distance bank: [Batch, Out_Features]
            diff = x.unsqueeze(1) - self.W.unsqueeze(0)
            return -diff.abs().sum(dim=2)
        else:
            raise ValueError(f"Unknown proj_type: {self.proj_type}")

def train_learned_projection(statements_flat, num_classes, out_dim, epochs=100, lr=1e-3, proj_type="linear", device='cpu'):
    """
    Trains a projection layer to minimize L1 distance distortion + Triplet Margin Loss.
    """
    in_dim = statements_flat.shape[-1]
    model = LearnedProjection(in_dim, out_dim, proj_type=proj_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    statements_flat = statements_flat.to(device)
    num_statements = statements_flat.shape[0]
    labels = torch.arange(num_classes, device=device).repeat_interleave(10) # 10 statements per class
    
    orig_dist_scale = float(in_dim)
    proj_dist_scale = float(out_dim)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 1. Stochastic Pair Sampling for Distance Preservation
        idx1 = torch.randint(0, num_statements, (8192,), device=device)
        idx2 = torch.randint(0, num_statements, (8192,), device=device)
        
        x1 = statements_flat[idx1]
        x2 = statements_flat[idx2]
        
        # Pairwise original L1 distance (scaled to range [0, 1])
        d_orig = torch.norm(x1 - x2, p=1, dim=1) / orig_dist_scale
        
        # Projected L1 distance
        y1 = model(x1)
        y2 = model(x2)
        d_proj = torch.norm(y1 - y2, p=1, dim=1) / proj_dist_scale
        
        loss_dist = F.mse_loss(d_proj, d_orig)
        
        # 2. Triplet Loss for Margin & Neighborhood Preservation
        # Anchor: random statements
        anchor_idx = torch.randint(0, num_statements, (1024,), device=device)
        anchors = statements_flat[anchor_idx]
        anchor_labels = labels[anchor_idx]
        
        # Project all statements to compute centroids dynamically
        y_all = model(statements_flat) # [N_facts * 10, out_dim]
        y_reshaped = y_all.view(num_classes, 10, out_dim)
        centroids = y_reshaped.mean(dim=1) # [num_classes, out_dim]
        
        y_anchors = model(anchors) # [1024, out_dim]
        
        # Positive: centroid of the same class
        pos_centroids = centroids[anchor_labels] # [1024, out_dim]
        
        # Negative: centroid of a random different class
        neg_labels = (anchor_labels + torch.randint(1, num_classes, (1024,), device=device)) % num_classes
        neg_centroids = centroids[neg_labels] # [1024, out_dim]
        
        # L1 distances
        d_pos = torch.norm(y_anchors - pos_centroids, p=1, dim=1) / proj_dist_scale
        d_neg = torch.norm(y_anchors - neg_centroids, p=1, dim=1) / proj_dist_scale
        
        # Triplet margin loss (margin = 0.1 normalized)
        loss_triplet = torch.clamp(d_pos - d_neg + 0.1, min=0.0).mean()
        
        # Total combined loss
        loss = loss_dist + 2.0 * loss_triplet
        
        loss.backward()
        optimizer.step()
        
    return model

# ==============================================================================
# 4. EVALUATION MODULE
# ==============================================================================

def evaluate_projection(statements_emb, queries_emb, facts, projection_type, model=None, W_linear=None, statements_mean=None, V_W=None, R=None, original_centroids=None, device='cpu'):
    """
    Evaluates a projection method using retrieval and telemetry metrics.
    """
    statements_emb = statements_emb.to(device)
    queries_emb = queries_emb.to(device)
    centroids_384 = statements_emb.mean(dim=1)
    
    # Apply projection mapping
    if projection_type == "raw":
        centroids = centroids_384
        queries = queries_emb
        out_dim = 384
    elif projection_type == "random":
        R = R.to(device)
        centroids = centroids_384 @ R.T
        queries = queries_emb @ R.T
        out_dim = R.shape[0]
    elif projection_type == "svd":
        statements_mean = statements_mean.to(device)
        V_W = V_W.to(device)
        centroids = (centroids_384 - statements_mean) @ V_W.T
        queries = (queries_emb - statements_mean) @ V_W.T
        out_dim = V_W.shape[0]
    elif projection_type == "learned":
        model.eval()
        with torch.no_grad():
            centroids = model(centroids_384)
            queries = model(queries_emb)
        out_dim = model.out_features
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")
        
    # Calculate negative L1 pairwise distance
    dists = torch.cdist(queries.unsqueeze(0), centroids.unsqueeze(0), p=1).squeeze(0) # [N_queries, N_centroids]
    N = len(facts)
    sorted_indices = dists.argsort(dim=1)
    
    ranks = []
    margins = []
    gap_ratios = []
    top1_dists = []
    top2_dists = []
    decision_gaps = []
    
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    
    for q_idx in range(N):
        # Use positional index as centroid index (not the absolute label),
        # because centroids are built from eval_statements_emb which is
        # sliced to match eval_facts ordering (centroid[i] ↔ eval_facts[i]).
        correct_idx = q_idx
        sorted_ind = sorted_indices[q_idx].tolist()
        
        rank = sorted_ind.index(correct_idx) + 1
        ranks.append(rank)
        
        if rank == 1:
            recall_at_1 += 1
        if rank <= 5:
            recall_at_5 += 1
        if rank <= 10:
            recall_at_10 += 1
            
        # Margin and Distance Telemetry
        same_class_dist = dists[q_idx, correct_idx].item()
        top1_dists.append(same_class_dist)
        
        # Nearest other class
        other_dists = dists[q_idx].clone()
        other_dists[correct_idx] = float('inf')
        nearest_other_dist = other_dists.min().item()
        
        top2_dists.append(nearest_other_dist)
        margin = nearest_other_dist - same_class_dist
        margins.append(margin)
        
        gap_ratio = nearest_other_dist / (same_class_dist + 1e-8)
        gap_ratios.append(gap_ratio)
        
        # Rank-based decision gap (Top-2 distance difference)
        sorted_dists = dists[q_idx].sort()[0]
        decision_gap = sorted_dists[1].item() - sorted_dists[0].item()
        decision_gaps.append(decision_gap)
        
    # Rank Preservation (Neighbor Recall@10)
    neighbor_recalls = []
    if original_centroids is not None and projection_type != "raw":
        original_centroids = original_centroids.to(device)
        orig_centroid_dists = torch.cdist(centroids_384.unsqueeze(0), original_centroids.unsqueeze(0), p=1).squeeze(0) # [N, N]
        proj_centroid_dists = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0), p=1).squeeze(0) # [N, N]
        
        orig_neighbors = orig_centroid_dists.argsort(dim=1)[:, 1:11]
        proj_neighbors = proj_centroid_dists.argsort(dim=1)[:, 1:11]
        
        for i in range(N):
            o_set = set(orig_neighbors[i].tolist())
            p_set = set(proj_neighbors[i].tolist())
            overlap = len(o_set.intersection(p_set))
            neighbor_recalls.append(overlap / 10.0)
    else:
        neighbor_recalls = [1.0] * N
        
    # Distance Distortion Rate
    distortions = []
    if original_centroids is not None and projection_type != "raw":
        original_centroids = original_centroids.to(device)
        orig_cdists = torch.cdist(centroids_384.unsqueeze(0), original_centroids.unsqueeze(0), p=1).squeeze(0) / 384.0
        proj_cdists = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0), p=1).squeeze(0) / float(out_dim)
        distortion_matrix = (orig_cdists - proj_cdists).abs()
        distortions.append(distortion_matrix.mean().item())
    else:
        distortions = [0.0]
        
    return {
        "recall_at_1": recall_at_1 / N,
        "recall_at_5": recall_at_5 / N,
        "recall_at_10": recall_at_10 / N,
        "mean_rank": np.mean(ranks),
        "median_rank": np.median(ranks),
        "mean_margin": np.mean(margins),
        "p5_margin": np.percentile(margins, 5),
        "p10_margin": np.percentile(margins, 10),
        "mean_gap_ratio": np.mean(gap_ratios),
        "mean_top1_dist": np.mean(top1_dists),
        "mean_top2_dist": np.mean(top2_dists),
        "mean_decision_gap": np.mean(decision_gaps),
        "neighbor_recall_at_10": np.mean(neighbor_recalls),
        "distortion_rate": distortions[0]
    }

# ==============================================================================
# 5. SWEEP ENGINE
# ==============================================================================

def run_sweep(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== Running Experiment 6 on {device.upper()} (Pilot: {args.pilot}) ===")
    
    results_dir = "."
    csv_path = os.path.join(results_dir, "kaggle_learned_projection_pilot.csv" if args.pilot else "kaggle_learned_projection.csv")
    plot_path = os.path.join(results_dir, "kaggle_learned_projection_pilot.png" if args.pilot else "kaggle_learned_projection.png")
    md_path = os.path.join(results_dir, "kaggle_learned_projection_pilot.md" if args.pilot else "kaggle_learned_projection.md")
    
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    if args.pilot:
        fact_sizes = [100, 400, 1600]
        widths = [32, 64, 128, 256]
    else:
        fact_sizes = [100, 200, 400, 800, 1600, 3200]
        widths = [32, 64, 128, 256]
        
    if args.N_max is not None:
        fact_sizes = [n for n in fact_sizes if n <= args.N_max]
        if not fact_sizes:
            fact_sizes = [args.N_max]
            
    seeds = [42, 101, 202]
    
    csv_fields = [
        "N", "mode", "projection_type", "W", "recall_at_1", "recall_at_5", "recall_at_10", 
        "mean_rank", "mean_margin", "p5_margin", "mean_gap_ratio", 
        "mean_top1_dist", "mean_top2_dist", "mean_decision_gap", "neighbor_recall_at_10", "distortion_rate"
    ]
    
    all_results = []
    
    for N in fact_sizes:
        print(f"\n[*] Generating and encoding {N} facts...")
        facts = generate_unambiguous_facts(N)
        
        all_statement_texts = [s for f in facts for s in f["statements"]]
        all_query_texts = [f["query"] for f in facts]
        
        with torch.no_grad():
            statements_flat = bi_encoder.encode(all_statement_texts, batch_size=256, convert_to_tensor=True, device=device).clone()
            statements_emb = statements_flat.view(N, 10, -1)
            queries_emb = bi_encoder.encode(all_query_texts, batch_size=256, convert_to_tensor=True, device=device).clone()
            
        original_centroids = statements_emb.mean(dim=1)
        
        for mode in ["capacity", "generalization"]:
            print(f"  --- Mode: {mode.upper()} ---")
            
            if mode == "capacity":
                eval_facts = facts
                eval_statements_emb = statements_emb
                eval_queries_emb = queries_emb
                eval_statements_flat = statements_flat
                eval_original_centroids = original_centroids
                
                statements_mean = statements_flat.mean(dim=0, keepdim=True)
                statements_centered = statements_flat - statements_mean
                _, _, Vh = torch.linalg.svd(statements_centered.to(device), full_matrices=False)
                Vh = Vh.cpu()
                
            else:
                train_size = N // 2
                if train_size < 1:
                    continue
                
                train_facts = facts[:train_size]
                eval_facts = facts[train_size:]
                
                train_statements_flat = statements_flat[:train_size * 10]
                eval_statements_emb = statements_emb[train_size:]
                eval_queries_emb = queries_emb[train_size:]
                eval_statements_flat = statements_flat[train_size * 10:]
                eval_original_centroids = original_centroids[train_size:]
                
                statements_mean = train_statements_flat.mean(dim=0, keepdim=True)
                statements_centered = train_statements_flat - statements_mean
                _, _, Vh = torch.linalg.svd(statements_centered.to(device), full_matrices=False)
                Vh = Vh.cpu()
            
            raw_res = evaluate_projection(eval_statements_emb, eval_queries_emb, eval_facts, "raw", device=device)
            raw_res["N"] = N
            raw_res["mode"] = mode
            raw_res["projection_type"] = "identity"
            raw_res["W"] = 384
            all_results.append(raw_res)
            print(f"    Identity: Recall@1={raw_res['recall_at_1']*100:.1f}%, Margin={raw_res['mean_margin']:.2f}")
            
            for W in widths:
                print(f"    W = {W}:")
                
                rand_recalls_at_1 = []
                rand_recalls_at_5 = []
                rand_recalls_at_10 = []
                rand_margins = []
                rand_neighbor_recalls = []
                rand_distortions = []
                rand_p5_margins = []
                rand_decision_gaps = []
                
                for seed in seeds:
                    torch.manual_seed(seed)
                    R = torch.randn(W, 384)
                    R = R / (R.norm(p=2, dim=1, keepdim=True) + 1e-8)
                    
                    res = evaluate_projection(eval_statements_emb, eval_queries_emb, eval_facts, "random", R=R, original_centroids=eval_original_centroids, device=device)
                    rand_recalls_at_1.append(res["recall_at_1"])
                    rand_recalls_at_5.append(res["recall_at_5"])
                    rand_recalls_at_10.append(res["recall_at_10"])
                    rand_margins.append(res["mean_margin"])
                    rand_p5_margins.append(res["p5_margin"])
                    rand_neighbor_recalls.append(res["neighbor_recall_at_10"])
                    rand_distortions.append(res["distortion_rate"])
                    rand_decision_gaps.append(res["mean_decision_gap"])
                    
                random_res = {
                    "recall_at_1": np.mean(rand_recalls_at_1),
                    "recall_at_5": np.mean(rand_recalls_at_5),
                    "recall_at_10": np.mean(rand_recalls_at_10),
                    "mean_rank": 0.0,
                    "median_rank": 0.0,
                    "mean_margin": np.mean(rand_margins),
                    "p5_margin": np.mean(rand_p5_margins),
                    "p10_margin": np.mean(rand_p5_margins),
                    "mean_gap_ratio": 0.0,
                    "mean_top1_dist": 0.0,
                    "mean_top2_dist": 0.0,
                    "mean_decision_gap": np.mean(rand_decision_gaps),
                    "neighbor_recall_at_10": np.mean(rand_neighbor_recalls),
                    "distortion_rate": np.mean(rand_distortions),
                    "N": N,
                    "mode": mode,
                    "projection_type": "random",
                    "W": W
                }
                all_results.append(random_res)
                print(f"      Random:   Recall@1={random_res['recall_at_1']*100:.1f}%, Margin={random_res['mean_margin']:.2f}")
                
                V_W = Vh[:W]
                svd_res = evaluate_projection(
                    eval_statements_emb, eval_queries_emb, eval_facts, "svd", 
                    statements_mean=statements_mean, V_W=V_W, original_centroids=eval_original_centroids, device=device
                )
                svd_res["N"] = N
                svd_res["mode"] = mode
                svd_res["projection_type"] = "svd"
                svd_res["W"] = W
                all_results.append(svd_res)
                print(f"      SVD:      Recall@1={svd_res['recall_at_1']*100:.1f}%, Margin={svd_res['mean_margin']:.2f}")
                
                epochs_to_train = 25 if args.pilot else 100
                if mode == "capacity":
                    learned_model = train_learned_projection(statements_flat, N, W, epochs=epochs_to_train, lr=1e-3, device=device)
                else:
                    learned_model = train_learned_projection(train_statements_flat, train_size, W, epochs=epochs_to_train, lr=1e-3, device=device)
                    
                learned_res = evaluate_projection(
                    eval_statements_emb, eval_queries_emb, eval_facts, "learned",
                    model=learned_model, original_centroids=eval_original_centroids, device=device
                )
                learned_res["N"] = N
                learned_res["mode"] = mode
                learned_res["projection_type"] = "learned"
                learned_res["W"] = W
                all_results.append(learned_res)
                print(f"      Learned:  Recall@1={learned_res['recall_at_1']*100:.1f}%, Margin={learned_res['mean_margin']:.2f}")
                
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in all_results:
            row = {field: r.get(field, 0.0) for field in csv_fields}
            writer.writerow(row)
            
    print(f"[*] Results saved to {csv_path}")
    generate_plots(all_results, plot_path)
    generate_report(all_results, md_path, args.pilot)
    print(f"[*] Report and plots generated.")

# ==============================================================================
# 6. PLOTTING & REPORT GENERATION
# ==============================================================================

def generate_plots(results, plot_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    modes = ["capacity", "generalization"]
    metrics = ["recall_at_1", "neighbor_recall_at_10"]
    titles = {
        ("capacity", "recall_at_1"): "Mode A (Capacity): Recall@1",
        ("capacity", "neighbor_recall_at_10"): "Mode A (Capacity): Neighbor Recall@10",
        ("generalization", "recall_at_1"): "Mode B (Generalization): Recall@1",
        ("generalization", "neighbor_recall_at_10"): "Mode B (Generalization): Neighbor Recall@10"
    }
    
    colors = {"identity": "black", "random": "red", "svd": "blue", "learned": "green"}
    markers = {"identity": "o", "random": "x", "svd": "^", "learned": "s"}
    
    Ns = sorted(list(set(r["N"] for r in results)))
    Ws = sorted(list(set(r["W"] for r in results if r["W"] != 384)))
    
    target_W = 128
    
    for row_idx, mode in enumerate(modes):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            ax.set_title(titles[(mode, metric)])
            ax.set_xlabel("N (Number of Facts)")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_xscale("log")
            
            id_data = [r for r in results if r["mode"] == mode and r["projection_type"] == "identity"]
            id_data = sorted(id_data, key=lambda x: x["N"])
            if id_data:
                ax.plot([d["N"] for d in id_data], [d[metric] for d in id_data], label="Identity (384D)", color=colors["identity"], marker=markers["identity"], linestyle="--")
                
            for proj in ["random", "svd", "learned"]:
                proj_data = [r for r in results if r["mode"] == mode and r["projection_type"] == proj and r["W"] == target_W]
                proj_data = sorted(proj_data, key=lambda x: x["N"])
                if proj_data:
                    ax.plot([d["N"] for d in proj_data], [d[metric] for d in proj_data], label=f"{proj.capitalize()} (W={target_W})", color=colors[proj], marker=markers[proj])
                    
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Second figure: Recall@1 vs Width at the largest available N, for both modes
    largest_N = max(r["N"] for r in results)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(f"Recall@1 vs Bottleneck Width (N={largest_N})")

    for col_idx, mode in enumerate(["capacity", "generalization"]):
        ax = axes2[col_idx]
        ax.set_title(f"Mode {'A (Capacity)' if mode == 'capacity' else 'B (Generalization)'}")
        ax.set_xlabel("Bottleneck Width (W)")
        ax.set_ylabel("Recall@1")

        for proj in ["random", "svd", "learned"]:
            proj_data = [r for r in results if r["mode"] == mode and r["projection_type"] == proj and r["N"] == largest_N]
            proj_data = sorted(proj_data, key=lambda x: x["W"])
            if proj_data:
                ax.plot([d["W"] for d in proj_data], [d["recall_at_1"] for d in proj_data],
                        label=f"{proj.capitalize()}", color=colors[proj], marker=markers[proj])

        # Identity baseline as a horizontal line (width-independent)
        id_data = [r for r in results if r["mode"] == mode and r["projection_type"] == "identity" and r["N"] == largest_N]
        if id_data:
            id_val = id_data[0]["recall_at_1"]
            ax.axhline(id_val, color=colors["identity"], linestyle="--", label=f"Identity (384D) = {id_val*100:.1f}%")

        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    width_plot_path = plot_path.replace(".png", "_width_scaling.png")
    plt.savefig(width_plot_path, dpi=200)
    plt.close()
    print(f"[*] Width-scaling plot saved to {width_plot_path}")

def generate_report(results, md_path, is_pilot):
    summary = defaultdict(dict)
    for r in results:
        key = (r["mode"], r["N"], r["W"])
        summary[key][r["projection_type"]] = r
        
    with open(md_path, "w") as f:
        f.write(f"# Experiment 6: Learned & SVD Projections Audit Report ({'Pilot' if is_pilot else 'Full Sweep'})\n\n")
        f.write("## 1. Key Performance Summary (Bottleneck W=128)\n\n")
        
        f.write("| Mode | N | Identity (384D) | Random (W=128) | SVD (W=128) | Learned (W=128) | Neighbor Recall (Learned) | L1 Distortion (Learned) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        modes = sorted(list(set(r["mode"] for r in results)))
        Ns = sorted(list(set(r["N"] for r in results)))
        
        for mode in modes:
            for N in Ns:
                key = (mode, N, 128)
                if key in summary:
                    id_r = summary[(mode, N, 384)].get("identity", {})
                    rand_r = summary[key].get("random", {})
                    svd_r = summary[key].get("svd", {})
                    learned_r = summary[key].get("learned", {})
                    
                    id_rec = f"{id_r.get('recall_at_1', 0.0)*100:.1f}%" if id_r else "-"
                    rand_rec = f"{rand_r.get('recall_at_1', 0.0)*100:.1f}%" if rand_r else "-"
                    svd_rec = f"{svd_r.get('recall_at_1', 0.0)*100:.1f}%" if svd_r else "-"
                    learned_rec = f"{learned_r.get('recall_at_1', 0.0)*100:.1f}%" if learned_r else "-"
                    
                    neigh_rec = f"{learned_r.get('neighbor_recall_at_10', 0.0)*100:.1f}%" if learned_r else "-"
                    dist_dist = f"{learned_r.get('distortion_rate', 0.0):.4f}" if learned_r else "-"
                    
                    f.write(f"| {mode.capitalize()} | {N} | {id_rec} | {rand_rec} | {svd_rec} | {learned_rec} | {neigh_rec} | {dist_dist} |\n")
                    
        f.write("\n## 2. Margin & Telemetry Details (W=128)\n\n")
        f.write("| Mode | N | Baseline | Margin (Mean) | Margin (P5) | Decision Gap | \n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for mode in modes:
            for N in Ns:
                key = (mode, N, 128)
                if key in summary:
                    for proj in ["identity", "random", "svd", "learned"]:
                        proj_key = (mode, N, 384) if proj == "identity" else key
                        r = summary[proj_key].get(proj, {})
                        if r:
                            f.write(f"| {mode.capitalize()} | {N} | {proj.capitalize()} | {r.get('mean_margin', 0.0):.2f} | {r.get('p5_margin', 0.0):.2f} | {r.get('mean_decision_gap', 0.0):.2f} |\n")

# ==============================================================================
# 7. MULTI-SEED CORPUS VALIDATION
# ==============================================================================

def run_multiseed_validation(args):
    """
    Targeted robustness check: N=3200, W=128, both modes, across multiple corpus
    generation seeds. Reports mean ± std for Recall@1 and mean_margin side by side
    for Identity, SVD, Learned, and Random projections.

    This is the experiment that turns promising single-corpus observations into
    defensible claims. If trends hold here they are real; if they vary strongly
    by corpus seed, that is equally important to know.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = args.ms_N
    W = args.ms_W
    corpus_seeds = list(range(args.ms_corpus_seeds))
    proj_random_seeds = [42, 101, 202]  # kept fixed — only corpus generation varies

    print(f"\n{'='*65}")
    print(f"  MULTI-SEED VALIDATION  |  N={N}  W={W}  corpus_seeds={len(corpus_seeds)}")
    print(f"{'='*65}")

    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # Structure: results[mode][proj] = list of (recall_at_1, mean_margin) per corpus seed
    projections = ["identity", "random", "svd", "learned"]
    modes = ["capacity", "generalization"]
    results = {mode: {proj: [] for proj in projections} for mode in modes}

    for corpus_seed in corpus_seeds:
        print(f"\n--- Corpus seed {corpus_seed} ---")

        # Generate a fresh fact corpus for this seed
        random.seed(corpus_seed)
        np.random.seed(corpus_seed)
        facts = generate_unambiguous_facts(N)

        all_statement_texts = [s for f in facts for s in f["statements"]]
        all_query_texts = [f["query"] for f in facts]

        with torch.no_grad():
            statements_flat = bi_encoder.encode(
                all_statement_texts, batch_size=256, convert_to_tensor=True, device=device
            ).clone()
            statements_emb = statements_flat.view(N, 10, -1)
            queries_emb = bi_encoder.encode(
                all_query_texts, batch_size=256, convert_to_tensor=True, device=device
            ).clone()

        original_centroids = statements_emb.mean(dim=1)

        for mode in modes:
            if mode == "capacity":
                eval_facts          = facts
                eval_statements_emb = statements_emb
                eval_queries_emb    = queries_emb
                eval_statements_flat = statements_flat
                eval_original_centroids = original_centroids

                svd_source = statements_flat
            else:
                train_size = N // 2
                eval_facts          = facts[train_size:]
                eval_statements_emb = statements_emb[train_size:]
                eval_queries_emb    = queries_emb[train_size:]
                eval_statements_flat = statements_flat[train_size * 10:]
                eval_original_centroids = original_centroids[train_size:]

                svd_source = statements_flat[:train_size * 10]

            # SVD fit on the appropriate split
            statements_mean = svd_source.mean(dim=0, keepdim=True)
            statements_centered = svd_source - statements_mean
            _, _, Vh = torch.linalg.svd(statements_centered.to(device), full_matrices=False)
            Vh = Vh.cpu()
            V_W = Vh[:W]

            # ---- Identity ----
            res = evaluate_projection(
                eval_statements_emb, eval_queries_emb, eval_facts, "raw", device=device
            )
            results[mode]["identity"].append((res["recall_at_1"], res["mean_margin"]))
            print(f"  [{mode:<15}] Identity : Recall@1={res['recall_at_1']*100:.1f}%  Margin={res['mean_margin']:.3f}")

            # ---- Random (averaged over proj_random_seeds) ----
            rand_r1s, rand_margins = [], []
            for rseed in proj_random_seeds:
                torch.manual_seed(rseed)
                R = torch.randn(W, 384)
                R = R / (R.norm(p=2, dim=1, keepdim=True) + 1e-8)
                res = evaluate_projection(
                    eval_statements_emb, eval_queries_emb, eval_facts,
                    "random", R=R, original_centroids=eval_original_centroids, device=device
                )
                rand_r1s.append(res["recall_at_1"])
                rand_margins.append(res["mean_margin"])
            rand_r1   = float(np.mean(rand_r1s))
            rand_marg = float(np.mean(rand_margins))
            results[mode]["random"].append((rand_r1, rand_marg))
            print(f"  [{mode:<15}] Random   : Recall@1={rand_r1*100:.1f}%  Margin={rand_marg:.3f}")

            # ---- SVD ----
            res = evaluate_projection(
                eval_statements_emb, eval_queries_emb, eval_facts, "svd",
                statements_mean=statements_mean, V_W=V_W,
                original_centroids=eval_original_centroids, device=device
            )
            results[mode]["svd"].append((res["recall_at_1"], res["mean_margin"]))
            print(f"  [{mode:<15}] SVD      : Recall@1={res['recall_at_1']*100:.1f}%  Margin={res['mean_margin']:.3f}")

            # ---- Learned ----
            if mode == "capacity":
                learned_model = train_learned_projection(
                    statements_flat, N, W, epochs=100, lr=1e-3, device=device
                )
            else:
                learned_model = train_learned_projection(
                    svd_source, N // 2, W, epochs=100, lr=1e-3, device=device
                )
            res = evaluate_projection(
                eval_statements_emb, eval_queries_emb, eval_facts, "learned",
                model=learned_model, original_centroids=eval_original_centroids, device=device
            )
            results[mode]["learned"].append((res["recall_at_1"], res["mean_margin"]))
            print(f"  [{mode:<15}] Learned  : Recall@1={res['recall_at_1']*100:.1f}%  Margin={res['mean_margin']:.3f}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    csv_path = f"kaggle_multiseed_N{N}_W{W}.csv"
    md_path  = f"kaggle_multiseed_N{N}_W{W}.md"

    csv_rows = []
    print(f"\n{'='*65}")
    print(f"  SUMMARY  (N={N}, W={W}, {len(corpus_seeds)} corpus seeds)")
    print(f"{'='*65}")

    for mode in modes:
        print(f"\n  Mode: {mode.upper()}")
        print(f"  {'Projection':<12} {'Recall@1 mean±std':>20}  {'Margin mean±std':>18}")
        print(f"  {'-'*54}")
        for proj in projections:
            vals = results[mode][proj]
            r1s     = [v[0] for v in vals]
            margins = [v[1] for v in vals]
            r1_mean, r1_std     = np.mean(r1s)*100,     np.std(r1s)*100
            marg_mean, marg_std = np.mean(margins),     np.std(margins)
            print(f"  {proj:<12}  {r1_mean:5.1f}% ± {r1_std:4.1f}%          {marg_mean:+.3f} ± {marg_std:.3f}")
            csv_rows.append({
                "N": N, "W": W, "mode": mode, "projection": proj,
                "corpus_seeds": len(corpus_seeds),
                "recall_at_1_mean": round(r1_mean / 100, 4),
                "recall_at_1_std":  round(r1_std  / 100, 4),
                "margin_mean": round(marg_mean, 4),
                "margin_std":  round(marg_std,  4),
            })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[*] CSV saved to {csv_path}")

    # Markdown report
    with open(md_path, "w") as f:
        f.write(f"# Multi-Seed Validation: N={N}, W={W}\n\n")
        f.write(f"**Corpus seeds:** {len(corpus_seeds)}  |  **Random projection seeds:** {proj_random_seeds}\n\n")
        for mode in modes:
            f.write(f"## Mode: {mode.capitalize()}\n\n")
            f.write("| Projection | Recall@1 mean ± std | Margin mean ± std |\n")
            f.write("| :--- | :---: | :---: |\n")
            for proj in projections:
                vals = results[mode][proj]
                r1s     = [v[0] for v in vals]
                margins = [v[1] for v in vals]
                r1_mean, r1_std     = np.mean(r1s)*100, np.std(r1s)*100
                marg_mean, marg_std = np.mean(margins), np.std(margins)
                f.write(f"| {proj.capitalize()} | {r1_mean:.1f}% ± {r1_std:.1f}% | {marg_mean:+.3f} ± {marg_std:.3f} |\n")
            f.write("\n")
    print(f"[*] Report saved to {md_path}")


# ==============================================================================
# 8. MAIN ENTRY — runs everything automatically, no flags needed
# ==============================================================================

class _Args:
    """Hardcoded config — edit these values if you want to change anything."""
    # Full sweep settings
    pilot  = False   # False = full sweep (N up to 3200). Set True for a quick test.
    N_max  = None    # Leave as None for the full run.
    # Multi-seed validation settings
    ms_N            = 3200  # Which N to stress-test
    ms_W            = 128   # Which width to stress-test
    ms_corpus_seeds = 5     # How many independent fact corpora to generate

if __name__ == "__main__":
    args = _Args()
    set_seed(42)

    print("\n" + "="*65)
    print("  STEP 1 OF 2: Full sweep across all N and W")
    print("="*65)
    run_sweep(args)

    print("\n" + "="*65)
    print("  STEP 2 OF 2: Multi-seed robustness validation")
    print("="*65)
    run_multiseed_validation(args)