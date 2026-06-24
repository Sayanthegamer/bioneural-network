import sys
import os
import csv
import time
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Add mnc_project to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'mnc_project'))
from mnc.layers import MNCLinear
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

def generate_facts(N, collision_mode=False):
    facts = []
    random_gen = random.Random(12345)
    
    # Vocabulary lists for collision mode
    rooms = ["vault", "lobby", "lab", "archive", "server", "office", "warehouse", "depot", "hangar", "observatory", "basement", "attic", "corridor", "lounge", "cafeteria"]
    entrances = ["main door", "side entrance", "emergency exit", "rear gate", "elevator door", "service hatch", "loading dock", "security gate", "roof hatch", "ventilation shaft"]
    
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "black", "white", "gray", "pink", "cyan", "magenta", "teal", "indigo"]
    projects = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron"]
    locations = ["cabinet A", "cabinet B", "drawer 1", "drawer 2", "shelf X", "shelf Y", "safe box", "storage bin", "desk tray", "locker 9"]
    
    coordinators = ["Sarah", "David", "Emma", "James", "Sophia", "Daniel", "Olivia", "Michael", "Isabella", "William"]
    sessions = ["sync", "review", "planning", "alignment", "retrospective", "briefing", "workshop", "interview", "debrief", "consultation"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    hours = ["9 AM", "10 AM", "11 AM", "1 PM", "2 PM", "3 PM", "4 PM", "5 PM"]

    for i in range(N):
        num = random_gen.randint(1000, 9999)
        template_type = i % 3
        
        if collision_mode:
            # Shared names/entities (creates semantic collisions)
            room = rooms[i % len(rooms)]
            entrance = entrances[(i // len(rooms)) % len(entrances)]
            
            color = colors[i % len(colors)]
            project = projects[(i // len(colors)) % len(projects)]
            loc = locations[(i // (len(colors) * len(projects))) % len(locations)]
            
            coord = coordinators[i % len(coordinators)]
            session = sessions[(i // len(coordinators)) % len(sessions)]
            day = days[(i // (len(coordinators) * len(sessions))) % len(days)]
            hour = hours[(i // (len(coordinators) * len(sessions) * len(days))) % len(hours)]
        else:
            # Unique names/entities using index i
            room = f"room {i}"
            entrance = f"entrance {i}"
            color = f"color {i}"
            project = f"project {i}"
            loc = f"location {i}"
            coord = f"coordinator {i}"
            session = f"session {i}"
            day = f"day {i}"
            hour = f"{i % 12 + 1} PM"
            
        statements = []
        if template_type == 0:
            statements.append(f"The access code for the {room} {entrance} is {num}.")
            statements.append(f"Entering {room} {entrance} requires keying in {num}.")
            statements.append(f"To unlock the door to {room} {entrance}, type in {num}.")
            statements.append(f"The digit sequence {num} grants entry to {room} {entrance}.")
            statements.append(f"Passcode {num} is configured for access to {room} {entrance}.")
            statements.append(f"You must enter keycode {num} to access {room} {entrance}.")
            statements.append(f"The entry pin for the {room} {entrance} is registered as {num}.")
            statements.append(f"Use password {num} when entering the {room} {entrance}.")
            statements.append(f"To get into {room} {entrance}, type the code {num} on the keypad.")
            statements.append(f"The lock on {room} {entrance} opens with passcode {num}.")
            
            query = f"Which combination of numbers unlocks the {entrance} of the {room}?"
            
        elif template_type == 1:
            statements.append(f"The {color} folder for Project {project} is in {loc}.")
            statements.append(f"You can find the Project {project} paperwork of color {color} inside {loc}.")
            statements.append(f"The {loc} contains the {color}-colored documents for Project {project}.")
            statements.append(f"I stored the Project {project} file colored {color} in {loc}.")
            statements.append(f"The {color} Project {project} folder is resting inside {loc}.")
            statements.append(f"Go check {loc} for the {color} folder belonging to Project {project}.")
            statements.append(f"The {color} file for Project {project} is kept in {loc}.")
            statements.append(f"Project {project} has its {color} dossier placed in {loc}.")
            statements.append(f"Inside {loc}, you will locate the {color} folder for Project {project}.")
            statements.append(f"The {color} folder representing Project {project} was left in {loc}.")
            
            query = f"Where should I look for the documents associated with the {color} Project {project}?"
            
        else:
            statements.append(f"Sarah's meeting for task {i} is scheduled on Tuesday at {hour} PM." if not collision_mode else f"{coord}'s {session} meeting is {day} at {hour}.")
            statements.append(f"On Tuesday at {hour} PM, Sarah will host a sync regarding task {i}." if not collision_mode else f"On {day} at {hour}, {coord} will host a {session} session.")
            statements.append(f"The calendar invitation for task {i} with Sarah is set for Tuesday at {hour} PM." if not collision_mode else f"The calendar invitation for the {session} with {coord} is set for {day} at {hour}.")
            statements.append(f"At {hour} PM this coming Tuesday, Sarah is running a session on task {i}." if not collision_mode else f"At {hour} on {day}, {coord} is running a {session}.")
            statements.append(f"Sarah will discuss the status of task {i} at {hour} PM on Tuesday." if not collision_mode else f"{coord} will discuss the status of {session} at {hour} on {day}.")
            statements.append(f"The meeting about task {i} with Sarah is slated for Tuesday at {hour} PM." if not collision_mode else f"The meeting about {session} with {coord} is slated for {day} at {hour}.")
            statements.append(f"Sarah's calendar shows a discussion for task {i} on Tuesday at {hour} PM." if not collision_mode else f"{coord}'s calendar shows a discussion for {session} on {day} at {hour}.")
            statements.append(f"We have a scheduling for task {i} with Sarah on Tuesday at {hour} PM." if not collision_mode else f"We have a scheduling for {session} with {coord} on {day} at {hour}.")
            statements.append(f"Tuesday at {hour} PM is when Sarah's meeting on task {i} takes place." if not collision_mode else f"{day} at {hour} is when {coord}'s meeting on {session} takes place.")
            statements.append(f"Sarah is holding the status sync for task {i} Tuesday at {hour} PM." if not collision_mode else f"{coord} is holding the status sync for {session} {day} at {hour}.")
            
            query = f"When does the meeting for task {i} scheduled?" if not collision_mode else f"When does the {session} session with {coord} start on {day}?"
            
        facts.append({"statements": statements, "query": query, "label": i})
    return facts

def kmeans_pytorch(x, k, num_iters=10):
    num_samples = x.size(0)
    if num_samples <= k:
        return x, torch.arange(num_samples, device=x.device)
    
    # Deterministic start: pick first k elements
    centroids = x[:k].clone()
    
    for _ in range(num_iters):
        diff = x.unsqueeze(1) - centroids.unsqueeze(0)
        dists = diff.abs().sum(dim=2)  # L1
        labels = dists.argmin(dim=1)
        
        new_centroids = centroids.clone()
        for c in range(k):
            mask = (labels == c)
            if mask.sum() > 0:
                new_centroids[c] = x[mask].mean(dim=0)
        centroids = new_centroids
        
    return centroids, labels

def compute_silhouette_score_l1(x, labels):
    num_samples = x.size(0)
    unique_labels = torch.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
        
    silhouettes = []
    for i in range(num_samples):
        c_i = labels[i].item()
        same_cluster = x[labels == c_i]
        other_cluster = x[labels != c_i]
        
        if same_cluster.size(0) > 1:
            a_i = (same_cluster - x[i].unsqueeze(0)).abs().sum(dim=1).sum().item() / (same_cluster.size(0) - 1)
        else:
            a_i = 0.0
            
        if other_cluster.size(0) > 0:
            b_i = (other_cluster - x[i].unsqueeze(0)).abs().sum(dim=1).mean().item()
        else:
            b_i = 0.0
            
        max_val = max(a_i, b_i)
        if max_val > 1e-8:
            silhouettes.append((b_i - a_i) / max_val)
        else:
            silhouettes.append(0.0)
            
    return np.mean(silhouettes)

def main():
    print("====================================================")
    print("  Experiment 2: Breaking the Prototype Assumption")
    print("====================================================")
    
    pipeline = JournalPipeline()
    seeds = [42, 101]
    
    # Sweeps
    N_list = [5, 10, 20, 50, 100, 200, 400, 800]
    spaces = [
        {"name": "raw_384d", "dim": 384, "project": False},
        {"name": "proj_128d", "dim": 128, "project": True},
        {"name": "proj_256d", "dim": 256, "project": True}
    ]
    
    results_dir = "experiments/results/experiment_2_prototype_assumption"
    logs_dir = "experiments/results/diagnostic_telemetry/logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "break_prototype_assumption.csv")
    headers = [
        "collision_mode", "space", "N", "seed", "method", "recall", "recall_per_vector",
        "mean_delta", "mean_margin", "p5_margin", "p1_margin",
        "mean_class_dispersion", "mean_intra_dist", "mean_inter_dist",
        "mean_lcomp_ratio", "mean_proto_bias", "mean_silhouette"
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
    for collision_mode in [False, True]:
        col_str = "RELATIONAL" if collision_mode else "UNIQUE_ID"
        print(f"\n==========================================")
        print(f"  REGIME: {col_str} COLLISION MODE")
        print(f"==========================================")
        
        for N in N_list:
            print(f"\n[*] Fact count N = {N}...")
            facts = generate_facts(N, collision_mode=collision_mode)
            
            # Embed statements and queries
            print("  [-] Generating embeddings via MiniLM (Batched)...")
            with torch.no_grad():
                all_statements = [s for f in facts for s in f["statements"]]
                all_queries = [f["query"] for f in facts]
                
                statements_flat = pipeline.encoder.encode(all_statements, batch_size=256, convert_to_tensor=True, device='cpu')
                raw_statements = statements_flat.view(N, 10, 384)
                
                raw_queries = pipeline.encoder.encode(all_queries, batch_size=256, convert_to_tensor=True, device='cpu')
                
            for space_cfg in spaces:
                space_name = space_cfg["name"]
                dim = space_cfg["dim"]
                print(f"  [-] Evaluating space: {space_name} (dimension {dim})...")
                
                for seed in seeds:
                    set_seed(seed)
                    
                    # Project if needed
                    if space_cfg["project"]:
                        shift, scale = autocalibrate_scale_distances(dim)
                        mnc_linear = MNCLinear(384, dim)
                        for param in mnc_linear.parameters():
                            param.requires_grad = False
                            
                        # Apply projection to statements and queries
                        # statements: [N, 10, dim]
                        flat_statements = raw_statements.view(N * 10, 384)
                        proj_flat = mnc_linear(flat_statements)
                        proj_flat = (proj_flat + shift) / scale
                        proj_flat = torch.tanh(proj_flat)
                        statements = proj_flat.view(N, 10, dim)
                        
                        proj_queries = mnc_linear(raw_queries)
                        proj_queries = (proj_queries + shift) / scale
                        queries = torch.tanh(proj_queries)
                    else:
                        statements = raw_statements.clone()
                        queries = raw_queries.clone()
                        
                    # Pre-calculate centroids, sub-centroids, silhouettes
                    centroids = statements.mean(dim=1) # [N, dim]
                    
                    multi_centroids_2 = []
                    silhouettes_2 = []
                    for c in range(N):
                        cens, lbls = kmeans_pytorch(statements[c], k=2)
                        multi_centroids_2.append(cens) # [2, dim]
                        silhouettes_2.append(compute_silhouette_score_l1(statements[c], lbls))
                    multi_centroids_2 = torch.stack(multi_centroids_2) # [N, 2, dim]
                    mean_silhouette = np.mean(silhouettes_2)
                    
                    multi_centroids_3 = []
                    for c in range(N):
                        cens, _ = kmeans_pytorch(statements[c], k=3)
                        multi_centroids_3.append(cens) # [3, dim]
                    multi_centroids_3 = torch.stack(multi_centroids_3) # [N, 3, dim]
                    
                    # Geometry metrics
                    # Dispersion
                    dispersions = []
                    for c in range(N):
                        disp = (statements[c] - centroids[c].unsqueeze(0)).abs().sum(dim=1).mean().item()
                        dispersions.append(disp)
                    mean_dispersion = np.mean(dispersions)
                    
                    # Intra-class
                    intra_dists = []
                    for c in range(N):
                        diff_intra = statements[c].unsqueeze(1) - statements[c].unsqueeze(0)
                        dists_intra = diff_intra.abs().sum(dim=2)
                        # Extract upper triangle (excluding diagonal)
                        mask = torch.triu(torch.ones(10, 10), diagonal=1).bool()
                        intra_dists.append(dists_intra[mask].mean().item())
                    mean_intra_dist = np.mean(intra_dists)
                    
                    # Inter-class
                    diff_inter = centroids.unsqueeze(1) - centroids.unsqueeze(0)
                    dists_inter = diff_inter.abs().sum(dim=2)
                    mask_inter = torch.triu(torch.ones(N, N), diagonal=1).bool()
                    if N > 1:
                        mean_inter_dist = dists_inter[mask_inter].mean().item()
                    else:
                        mean_inter_dist = 0.0
                        
                    # Evaluate all retrieval methods
                    methods = ["Prototype", "1-NN", "k-NN_3", "k-NN_5", "Multi-Proto_2", "Multi-Proto_3", "Oracle"]
                    
                    for method in methods:
                        correct_count = 0
                        deltas = []
                        margins = []
                        lcomp_ratios = []
                        proto_biases = []
                        
                        # Pre-cache database for k-NN queries
                        # db_statements shape: [N * 10, dim]
                        db_statements = statements.view(N * 10, dim)
                        db_labels = torch.tensor([facts[c]["label"] for c in range(N) for _ in range(10)])
                        
                        for i in range(N):
                            q = queries[i] # [dim]
                            
                            # 1. Compute L1 distance to all individual statements in the database
                            # diff_all shape: [N * 10, dim]
                            diff_all = db_statements - q.unsqueeze(0)
                            dists_all = diff_all.abs().sum(dim=1) # [N * 10]
                            
                            # Group L1 distances by class
                            # class_dists shape: [N, 10]
                            class_dists = dists_all.view(N, 10)
                            
                            # Distances to centroids
                            # dists_proto shape: [N]
                            dists_proto = (centroids - q.unsqueeze(0)).abs().sum(dim=1)
                            
                            # Distances to sub-centroids
                            # multi_centroids_2 shape: [N, 2, dim]
                            dists_multi2 = (multi_centroids_2 - q.unsqueeze(0).unsqueeze(1)).abs().sum(dim=2).min(dim=1).values
                            dists_multi3 = (multi_centroids_3 - q.unsqueeze(0).unsqueeze(1)).abs().sum(dim=2).min(dim=1).values
                            
                            # Compute method prediction
                            if method == "Prototype":
                                pred_dists = dists_proto
                                pred_class = pred_dists.argmin().item()
                                
                            elif method == "1-NN":
                                pred_class = db_labels[dists_all.argmin().item()].item()
                                pred_dists = class_dists.min(dim=1).values
                                
                            elif method == "Oracle":
                                # Mathematically equivalent to 1-NN: predict class of closest exemplar
                                pred_dists = class_dists.min(dim=1).values
                                pred_class = pred_dists.argmin().item()
                                
                            elif method.startswith("k-NN"):
                                k = int(method.split("_")[1])
                                topk_idx = dists_all.argsort()[:k]
                                topk_labels = db_labels[topk_idx].tolist()
                                topk_dists = dists_all[topk_idx]
                                
                                # Distance weighted vote
                                weights = 1.0 / (topk_dists + 1e-5)
                                votes = {}
                                for idx_v, lbl in enumerate(topk_labels):
                                    votes[lbl] = votes.get(lbl, 0.0) + weights[idx_v].item()
                                pred_class = max(votes, key=votes.get)
                                
                                # Estimate surrogate distance for margin logging:
                                # We sum inverse weights to get distance proxy
                                pred_dists = torch.zeros(N)
                                for c in range(N):
                                    c_mask = (db_labels == c)
                                    pred_dists[c] = class_dists[c].min() # Fallback proxy
                                    
                            elif method == "Multi-Proto_2":
                                pred_dists = dists_multi2
                                pred_class = pred_dists.argmin().item()
                                
                            elif method == "Multi-Proto_3":
                                pred_dists = dists_multi3
                                pred_class = pred_dists.argmin().item()
                                
                            correct_label = facts[i]["label"]
                            if pred_class == correct_label:
                                correct_count += 1
                                
                            # Confidence delta: d_2nd_closest - d_1st_closest
                            sorted_dists = pred_dists.sort().values
                            delta = (sorted_dists[1] - sorted_dists[0]).item()
                            deltas.append(delta)
                            
                            # Class margin: d_nearest_incorrect - d_correct
                            d_correct = pred_dists[correct_label].item()
                            mask = torch.ones(N, dtype=torch.bool)
                            mask[correct_label] = False
                            d_wrong = pred_dists[mask].min().item()
                            margin = d_wrong - d_correct
                            margins.append(margin)
                            
                            # Compression metrics (Prototype specific logs)
                            closest_exemplar_dist = class_dists[correct_label].min().item()
                            lcomp = dists_proto[correct_label].item() / (closest_exemplar_dist + 1e-8)
                            lcomp_ratios.append(lcomp)
                            
                            proto_bias = dists_proto[correct_label].item() - class_dists[correct_label].mean().item()
                            proto_biases.append(proto_bias)
                            
                        # Aggregate statistics
                        recall = correct_count / float(N)
                        
                        # Memory multiplier adjustment
                        stored_vectors_map = {
                            "Prototype": 1.0,
                            "Oracle": 10.0,
                            "1-NN": 10.0,
                            "k-NN_3": 10.0,
                            "k-NN_5": 10.0,
                            "Multi-Proto_2": 2.0,
                            "Multi-Proto_3": 3.0
                        }
                        stored_vecs = stored_vectors_map[method]
                        recall_per_vector = recall / stored_vecs
                        
                        mean_delta = np.mean(deltas)
                        mean_margin = np.mean(margins)
                        p5_margin = np.percentile(margins, 5)
                        p1_margin = np.percentile(margins, 1)
                        mean_lcomp = np.mean(lcomp_ratios)
                        mean_pbias = np.mean(proto_biases)
                        
                        row = {
                            "collision_mode": collision_mode,
                            "space": space_name,
                            "N": N,
                            "seed": seed,
                            "method": method,
                            "recall": recall,
                            "recall_per_vector": recall_per_vector,
                            "mean_delta": mean_delta,
                            "mean_margin": mean_margin,
                            "p5_margin": p5_margin,
                            "p1_margin": p1_margin,
                            "mean_class_dispersion": mean_dispersion,
                            "mean_intra_dist": mean_intra_dist,
                            "mean_inter_dist": mean_inter_dist,
                            "mean_lcomp_ratio": mean_lcomp,
                            "mean_proto_bias": mean_pbias,
                            "mean_silhouette": mean_silhouette if "Multi-Proto" in method else 0.0
                        }
                        
                        with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=headers)
                            writer.writerow(row)
                            
    # ----------------------------------------------------
    # Auxiliary Metric Control Sweeps (L1 vs L2 vs Cosine)
    # ----------------------------------------------------
    print("\n==========================================")
    print("  AUXILIARY RETRIEVAL METRIC CONTROL (Raw Space)")
    print("==========================================")
    aux_csv_path = os.path.join(results_dir, "break_prototype_aux_control.csv")
    aux_headers = ["collision_mode", "N", "metric", "method", "recall"]
    with open(aux_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=aux_headers)
        writer.writeheader()
        
    for collision_mode in [False, True]:
        for N in [800, 1600]:
            print(f"  [-] Sweep N = {N} (collision={collision_mode})...")
            facts = generate_facts(N, collision_mode=collision_mode)
            
            with torch.no_grad():
                all_statements = [s for f in facts for s in f["statements"]]
                all_queries = [f["query"] for f in facts]
                
                statements_flat = pipeline.encoder.encode(all_statements, batch_size=256, convert_to_tensor=True, device='cpu')
                statements = statements_flat.view(N, 10, 384)
                
                queries = pipeline.encoder.encode(all_queries, batch_size=256, convert_to_tensor=True, device='cpu')
                
            centroids = statements.mean(dim=1)
            db_statements = statements.view(N * 10, 384)
            db_labels = torch.tensor([facts[c]["label"] for c in range(N) for _ in range(10)])
            
            metrics = ["L1", "L2", "Cosine"]
            methods = ["Prototype", "1-NN"]
            
            for metric in metrics:
                for method in methods:
                    correct_count = 0
                    for i in range(N):
                        q = queries[i]
                        
                        # Distance computation helper
                        if metric == "L1":
                            dists_all = (db_statements - q.unsqueeze(0)).abs().sum(dim=1)
                            dists_proto = (centroids - q.unsqueeze(0)).abs().sum(dim=1)
                        elif metric == "L2":
                            dists_all = torch.norm(db_statements - q.unsqueeze(0), p=2, dim=1)
                            dists_proto = torch.norm(centroids - q.unsqueeze(0), p=2, dim=1)
                        elif metric == "Cosine":
                            # Normalize
                            db_norm = db_statements / (db_statements.norm(p=2, dim=1, keepdim=True) + 1e-8)
                            q_norm = q / (q.norm(p=2) + 1e-8)
                            centroids_norm = centroids / (centroids.norm(p=2, dim=1, keepdim=True) + 1e-8)
                            # Cosine distance = 1 - sim
                            dists_all = 1.0 - torch.mm(db_norm, q_norm.unsqueeze(1)).squeeze(1)
                            dists_proto = 1.0 - torch.mm(centroids_norm, q_norm.unsqueeze(1)).squeeze(1)
                            
                        if method == "Prototype":
                            pred_class = dists_proto.argmin().item()
                        else:
                            pred_class = db_labels[dists_all.argmin().item()].item()
                            
                        if pred_class == facts[i]["label"]:
                            correct_count += 1
                            
                    recall = correct_count / float(N)
                    row = {
                        "collision_mode": collision_mode,
                        "N": N,
                        "metric": metric,
                        "method": method,
                        "recall": recall
                    }
                    with open(aux_csv_path, 'a', newline='', encoding='utf-8') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=aux_headers)
                        writer.writerow(row)
                        
    # ----------------------------------------------------
    # Generate Plots and Reports
    # ----------------------------------------------------
    plot_results(csv_path, results_dir)
    generate_report(csv_path, aux_csv_path, results_dir)

def plot_results(csv_path, results_dir):
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            data.append({
                "collision_mode": r["collision_mode"] == "True",
                "space": r["space"],
                "N": int(r["N"]),
                "method": r["method"],
                "recall": float(r["recall"])
            })
            
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    
    spaces_list = ["raw_384d", "proj_128d", "proj_256d"]
    for row_idx, col_mode in enumerate([False, True]):
        title_prefix = "Relational Collision" if col_mode else "Unique ID"
        for col_idx, space_name in enumerate(spaces_list):
            ax = axes[row_idx, col_idx]
            
            # Filter matches
            matches = [r for r in data if r["collision_mode"] == col_mode and r["space"] == space_name]
            methods = sorted(list(set([r["method"] for r in matches])))
            N_vals = sorted(list(set([r["N"] for r in matches])))
            
            for method in methods:
                rec_vals = []
                for n in N_vals:
                    sub = [r["recall"] for r in matches if r["method"] == method and r["N"] == n]
                    rec_vals.append(np.mean(sub) if sub else 0.0)
                ax.plot(N_vals, rec_vals, 'o-', label=method)
                
            ax.set_title(f"{title_prefix} | {space_name}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.0, 1.05)
            if row_idx == 1:
                ax.set_xlabel("Fact Horizon (N)")
            if col_idx == 0:
                ax.set_ylabel("Recall Accuracy")
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize='small')
                
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "break_prototype_assumption.png")
    plt.savefig(plot_path, dpi=150)
    print(f"[*] Saved plot to: {plot_path}")

def generate_report(csv_path, aux_csv_path, results_dir):
    # Parse data for summary
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            data.append(r)
            
    aux_data = []
    if os.path.exists(aux_csv_path):
        with open(aux_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                aux_data.append(r)
                
    md_path = os.path.join(results_dir, "break_prototype_assumption.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Experiment 2 Summary: Breaking the Prototype Assumption\n\n")
        f.write("## 🏆 Scientific Verdict\n\n")
        
        # Deconstruct critical results
        # We find N=400 and N=800 for Relational vs Unique ID
        f.write("### 1. Retrieval Comparison for N=400 and N=800\n\n")
        f.write("| Space | N | Regime | Prototype Recall | 1-NN Recall | Oracle Recall | Delta Margin (p5) | Lcomp Ratio | Silhouette |\n")
        f.write("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for space in ["raw_384d", "proj_128d", "proj_256d"]:
            for n_val in [400, 800]:
                for col_mode in ["False", "True"]:
                    col_name = "Relational" if col_mode == "True" else "Unique ID"
                    
                    # Fetch rows
                    proto_rows = [r for r in data if r["space"] == space and r["collision_mode"] == col_mode and int(r["N"]) == n_val and r["method"] == "Prototype"]
                    nn_rows = [r for r in data if r["space"] == space and r["collision_mode"] == col_mode and int(r["N"]) == n_val and r["method"] == "1-NN"]
                    oracle_rows = [r for r in data if r["space"] == space and r["collision_mode"] == col_mode and int(r["N"]) == n_val and r["method"] == "Oracle"]
                    
                    if proto_rows and nn_rows and oracle_rows:
                        p_recs = [float(r["recall"]) for r in proto_rows]
                        n_recs = [float(r["recall"]) for r in nn_rows]
                        o_recs = [float(r["recall"]) for r in oracle_rows]
                        p5_ms = [float(r["p5_margin"]) for r in proto_rows]
                        lcomps = [float(r["mean_lcomp_ratio"]) for r in proto_rows]
                        sils = [float(r["mean_silhouette"]) for r in proto_rows]
                        
                        p_rec = np.mean(p_recs) * 100.0
                        n_rec = np.mean(n_recs) * 100.0
                        o_rec = np.mean(o_recs) * 100.0
                        p5_m = np.mean(p5_ms)
                        lcomp = np.mean(lcomps)
                        sil = np.mean(sils)
                        f.write(f"| **{space}** | {n_val} | {col_name} | {p_rec:.1f}% | {n_rec:.1f}% | {o_rec:.1f}% | {p5_m:.4f} | {lcomp:.3f} | {sil:.3f} |\n")
                        
        f.write("\n### 2. Auxiliary Metric Control (Raw 384D Space, N=800 & 1600)\n\n")
        f.write("| N Facts | Regime | Metric | Prototype Recall | 1-NN Recall |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: |\n")
        for r in aux_data:
            col_name = "Relational" if r["collision_mode"] == "True" else "Unique ID"
            n_val = r["N"]
            met = r["metric"]
            meth = r["method"]
            rec = float(r["recall"]) * 100.0
            
            # Since rows are separate, we merge them for displaying
            # Let's just output raw rows
            f.write(f"| {n_val} | {col_name} | {met} | {meth} | {rec:.2f}% |\n")
            
        f.write("\n\n## 🔍 Interpretation & Recommendations\n")
        f.write("Evaluate the pre-hoc decision criteria:\n")
        f.write("1. **Is there retrieval compression loss?** Check the difference between Oracle/1-NN and Prototype recall. If it is < 10% and Lcomp is close to 1.0, then single-centroid prototype compression is not the main bottleneck.\n")
        f.write("2. **Is there a projection bottleneck?** Compare Raw 384D recall against Projected 128D/256D at identical fact horizon N. If the gap is > 10%, the random linear projection destroys manifold alignment.\n")
        f.write("3. **Is the manifold itself saturating?** If Raw 384D recall across all metrics/retrievals remains < 85% at N=800, the encoder's representational boundaries are overlapping.\n")
        
    print(f"[*] Saved markdown report to: {md_path}")

if __name__ == "__main__":
    main()
