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
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# Add mnc_project to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mnc_project'))
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

def generate_facts(N):
    facts = []
    random_gen = random.Random(12345)
    
    # Vocabulary lists for relational collisions
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
            statements.append(f"{coord}'s {session} meeting is {day} at {hour}.")
            statements.append(f"{coord} holds the {session} session on {day} scheduled for {hour}.")
            statements.append(f"The schedule lists {coord}'s {session} on {day} at {hour}.")
            statements.append(f"We meet with {coord} for the {session} on {day} at {hour}.")
            statements.append(f"At {hour} on {day}, {coord} conducts the {session} meeting.")
            statements.append(f"On {day} at {hour}, the {session} session with {coord} begins.")
            statements.append(f"{coord} runs the {session} gathering on {day} starting at {hour}.")
            statements.append(f"{coord}'s schedule includes the {session} on {day} at {hour}.")
            statements.append(f"{coord} will meet us for {session} on {day} at {hour}.")
            statements.append(f"The {session} with {coord} takes place on {day} at {hour}.")
            
            query = f"What is the day and hour of the {session} session hosted by {coord}?"
            
        facts.append({
            "label": i,
            "statements": statements,
            "query": query
        })
    return facts

def compute_distances(ref, query, metric):
    if metric == "L1":
        # ref: [N, D] or [N_all, D], query: [Q, D]
        return torch.cdist(query.unsqueeze(0), ref.unsqueeze(0), p=1).squeeze(0)
    elif metric == "L2":
        return torch.cdist(query.unsqueeze(0), ref.unsqueeze(0), p=2).squeeze(0)
    elif metric == "Cosine":
        # Cosine distance = 1 - sim
        ref_norm = ref / (ref.norm(p=2, dim=1, keepdim=True) + 1e-8)
        query_norm = query / (query.norm(p=2, dim=1, keepdim=True) + 1e-8)
        return 1.0 - torch.mm(query_norm, ref_norm.t())
    else:
        raise ValueError(f"Unknown metric {metric}")

def run_metric_evaluation(statements, queries, labels, metric, k_list=[1, 5, 10, 20]):
    # statements: [N, 10, D], queries: [N, D], labels: [N]
    N, num_exemplars, D = statements.shape
    
    centroids = statements.mean(dim=1) # [N, D]
    db_statements = statements.view(N * num_exemplars, D)
    db_labels = torch.tensor([i for i in range(N) for _ in range(num_exemplars)])
    
    # Pairwise distances from queries to class centroids and exemplars
    dists_proto = compute_distances(centroids, queries, metric) # [N_queries, N_centroids]
    dists_all = compute_distances(db_statements, queries, metric) # [N_queries, N_all_exemplars]
    
    # 1. Prototype and 1-NN recall
    correct_proto = 0
    correct_1nn = 0
    
    # Rank tracking
    proto_ranks = []
    exemplar_ranks = []
    
    # Oracles Top-K
    centroid_topk = {k: 0 for k in k_list}
    exemplar_topk = {k: 0 for k in k_list}
    
    # Margins and Lcomp
    margins = []
    lcomp_ratios = []
    
    for i in range(N):
        correct_label = i
        
        # Prototype evaluation
        p_dists = dists_proto[i]
        pred_proto = p_dists.argmin().item()
        if pred_proto == correct_label:
            correct_proto += 1
            
        # Rank of correct class in Prototype distance sorting
        sorted_p_classes = p_dists.argsort().tolist()
        p_rank = sorted_p_classes.index(correct_label)
        proto_ranks.append(p_rank)
        
        # Prototype Margin
        p_dist_correct = p_dists[correct_label].item()
        p_dist_wrong = p_dists[p_dists.argsort()[1]].item() if p_dists.argmin().item() == correct_label else p_dists[p_dists.argmin()].item()
        margin = p_dist_wrong - p_dist_correct
        margins.append(margin)
        
        # 1-NN evaluation
        e_dists = dists_all[i]
        pred_1nn = db_labels[e_dists.argmin().item()].item()
        if pred_1nn == correct_label:
            correct_1nn += 1
            
        # Rank of correct class in Exemplar distance sorting (minimum distance to class exemplars)
        # Compute min distance from query i to all classes
        class_min_dists = []
        for c in range(N):
            c_dists = e_dists[c * num_exemplars : (c + 1) * num_exemplars]
            class_min_dists.append(c_dists.min().item())
        class_min_dists = torch.tensor(class_min_dists)
        sorted_e_classes = class_min_dists.argsort().tolist()
        e_rank = sorted_e_classes.index(correct_label)
        exemplar_ranks.append(e_rank)
        
        # Oracles Top-K centroid
        for k in k_list:
            if correct_label in sorted_p_classes[:k]:
                centroid_topk[k] += 1
                
        # Oracles Top-K exemplar (is correct class in Top-K closest exemplars?)
        sorted_exemplar_labels = db_labels[e_dists.argsort()].tolist()
        # Find unique labels in order of closest exemplars
        unique_sorted_labels = []
        for lbl in sorted_exemplar_labels:
            if lbl not in unique_sorted_labels:
                unique_sorted_labels.append(lbl)
                if len(unique_sorted_labels) >= max(k_list):
                    break
        for k in k_list:
            if correct_label in unique_sorted_labels[:k]:
                exemplar_topk[k] += 1
                
        # Lcomp ratio
        closest_exemplar_dist = e_dists[correct_label * num_exemplars : (correct_label + 1) * num_exemplars].min().item()
        lcomp = p_dists[correct_label].item() / (closest_exemplar_dist + 1e-8)
        lcomp_ratios.append(lcomp)
        
    recall_proto = correct_proto / float(N)
    recall_1nn = correct_1nn / float(N)
    proto_advantage = recall_proto - recall_1nn
    
    centroid_topk_accs = {k: val / float(N) for k, val in centroid_topk.items()}
    exemplar_topk_accs = {k: val / float(N) for k, val in exemplar_topk.items()}
    
    mean_p_rank = np.mean(proto_ranks)
    median_p_rank = np.median(proto_ranks)
    p90_p_rank = np.percentile(proto_ranks, 90)
    
    mean_e_rank = np.mean(exemplar_ranks)
    median_e_rank = np.median(exemplar_ranks)
    p90_e_rank = np.percentile(exemplar_ranks, 90)
    
    # Margin percentiles
    margin_percentiles = {
        "p1": np.percentile(margins, 1),
        "p5": np.percentile(margins, 5),
        "p10": np.percentile(margins, 10),
        "p25": np.percentile(margins, 25),
        "p50": np.percentile(margins, 50),
        "p75": np.percentile(margins, 75),
        "p90": np.percentile(margins, 90)
    }
    
    # Normalized Separation Score (z-score)
    # Distance between class centroids
    if N > 1:
        inter_dists = compute_distances(centroids, centroids, metric)
        mask = ~torch.eye(N, dtype=torch.bool)
        inter_vals = inter_dists[mask].tolist()
        mu_inter = np.mean(inter_vals)
        sigma_inter = np.std(inter_vals) if len(inter_vals) > 1 else 1.0
    else:
        mu_inter = 0.0
        sigma_inter = 1.0
        
    # Distance of exemplars to their centroids
    intra_vals = []
    for c in range(N):
        c_dists = compute_distances(centroids[c].unsqueeze(0), statements[c], metric).squeeze(0).tolist()
        intra_vals.extend(c_dists)
    mu_intra = np.mean(intra_vals)
    
    z_score = (mu_inter - mu_intra) / (sigma_inter + 1e-8)
    
    return {
        "recall_proto": recall_proto,
        "recall_1nn": recall_1nn,
        "proto_advantage": proto_advantage,
        "centroid_topk": centroid_topk_accs,
        "exemplar_topk": exemplar_topk_accs,
        "mean_proto_rank": mean_p_rank,
        "median_proto_rank": median_p_rank,
        "p90_proto_rank": p90_p_rank,
        "mean_exemplar_rank": mean_e_rank,
        "median_exemplar_rank": median_e_rank,
        "p90_exemplar_rank": p90_e_rank,
        "margin_percentiles": margin_percentiles,
        "z_score": z_score,
        "mean_lcomp": np.mean(lcomp_ratios)
    }

def run_linear_probe(statements, queries, labels, N):
    num_ex = statements.shape[1]
    D = statements.shape[2]
    
    # Flatten statements for training
    X_train = statements.view(N * num_ex, D).cpu().numpy()
    y_train = np.array([i for i in range(N) for _ in range(num_ex)])
    
    X_test = queries.cpu().numpy()
    y_test = np.array([i for i in range(N)])
    
    # 1. Primary probe: LinearSVC
    clf_svc = LinearSVC(dual=False, C=1.0, random_state=42, max_iter=2000)
    clf_svc.fit(X_train, y_train)
    svc_acc = clf_svc.score(X_test, y_test)
    
    # 2. Secondary probe (cross-verification at N <= 400)
    lr_acc = -1.0
    if N <= 400:
        clf_lr = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42)
        clf_lr.fit(X_train, y_train)
        lr_acc = clf_lr.score(X_test, y_test)
        
    return svc_acc, lr_acc

def compute_jl_distortion(raw_vectors, proj_vectors, sample_size=5000):
    N_all = raw_vectors.shape[0]
    set_seed(42)
    
    idx_i = np.random.randint(0, N_all, size=sample_size)
    idx_j = np.random.randint(0, N_all, size=sample_size)
    
    valid_mask = (idx_i != idx_j)
    idx_i = idx_i[valid_mask]
    idx_j = idx_j[valid_mask]
    
    v_i_raw = raw_vectors[idx_i]
    v_j_raw = raw_vectors[idx_j]
    v_i_proj = proj_vectors[idx_i]
    v_j_proj = proj_vectors[idx_j]
    
    d_raw = torch.norm(v_i_raw - v_j_raw, p=1, dim=1)
    d_proj = torch.norm(v_i_proj - v_j_proj, p=1, dim=1)
    
    epsilons = torch.abs(d_proj / (d_raw + 1e-8) - 1.0).tolist()
    
    mean_dist = np.mean(epsilons)
    median_dist = np.median(epsilons)
    p95_dist = np.percentile(epsilons, 95)
    
    return mean_dist, median_dist, p95_dist

def main():
    print("====================================================")
    print("  Experiment 3: Relational Manifold Audit & Oracles")
    print("====================================================")
    
    encoders = [
        "all-MiniLM-L6-v2",
        "e5-small-v2",
        "all-mpnet-base-v2",
        "e5-base-v2",
        "bge-small-en-v1.5"
    ]
    
    encoder_mapping = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "e5-small-v2": "intfloat/e5-small-v2",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "e5-base-v2": "intfloat/e5-base-v2",
        "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5"
    }
    
    results_dir = "experiments/results"
    logs_dir = "experiments/results/logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "relational_manifold_audit.csv")
    headers = [
        "encoder", "norm_mode", "space", "projection", "N", "metric",
        "recall_proto", "recall_1nn", "proto_advantage", "mean_lcomp",
        "centroid_top1", "centroid_top5", "centroid_top10", "centroid_top20",
        "exemplar_top1", "exemplar_top5", "exemplar_top10", "exemplar_top20",
        "mean_proto_rank", "median_proto_rank", "p90_proto_rank",
        "mean_exemplar_rank", "median_exemplar_rank", "p90_exemplar_rank",
        "linear_probe_svc", "linear_probe_lr",
        "margin_p1", "margin_p5", "margin_p10", "margin_p25", "margin_p50", "margin_p75", "margin_p90",
        "z_score", "distortion_mean", "distortion_median", "distortion_p95"
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
    encoder_800_recalls = {}
    
    # --- FIRST PASS: Sweep all encoders up to N=800 ---
    for encoder_name in encoders:
        print(f"\n[+] Loading encoder: {encoder_name}...")
        pipeline = JournalPipeline(encoder_mapping[encoder_name])
        dim_in = pipeline.encoder.get_sentence_embedding_dimension()
        
        for N in [100, 200, 400, 800]:
            print(f"\n  [-] Running N = {N} facts...")
            facts = generate_facts(N)
            labels = torch.tensor([f["label"] for f in facts])
            
            is_e5 = "e5" in encoder_name.lower()
            print("      Ingesting text embeddings...")
            with torch.no_grad():
                if is_e5:
                    all_statements = [f"passage: {s}" for f in facts for s in f["statements"]]
                    all_queries = [f"query: {f['query']}" for f in facts]
                else:
                    all_statements = [s for f in facts for s in f["statements"]]
                    all_queries = [f["query"] for f in facts]
                
                statements_flat = pipeline.encoder.encode(all_statements, batch_size=256, convert_to_tensor=True, device='cpu')
                raw_statements = statements_flat.view(N, 10, dim_in)
                raw_queries = pipeline.encoder.encode(all_queries, batch_size=256, convert_to_tensor=True, device='cpu')
                
            for norm_mode in ["raw", "l2_normalized"]:
                print(f"      Normalization mode: {norm_mode}")
                if norm_mode == "l2_normalized":
                    norm_statements = raw_statements / (raw_statements.norm(p=2, dim=2, keepdim=True) + 1e-8)
                    norm_queries = raw_queries / (raw_queries.norm(p=2, dim=1, keepdim=True) + 1e-8)
                else:
                    norm_statements = raw_statements.clone()
                    norm_queries = raw_queries.clone()
                    
                spaces = [
                    {"name": "raw", "dim": dim_in, "project": False},
                    {"name": "proj_128d", "dim": 128, "project": True},
                    {"name": "proj_256d", "dim": 256, "project": True}
                ]
                
                for space_cfg in spaces:
                    space_name = space_cfg["name"]
                    dim_proj = space_cfg["dim"]
                    
                    proj_methods = ["none"]
                    if space_cfg["project"]:
                        proj_methods = ["random", "svd"]
                        
                    for proj_method in proj_methods:
                        print(f"      Space: {space_name} | Projection: {proj_method}")
                        
                        if proj_method == "random":
                            set_seed(42)
                            shift, scale = autocalibrate_scale_distances(dim_proj)
                            mnc_linear = MNCLinear(dim_in, dim_proj)
                            for param in mnc_linear.parameters():
                                param.requires_grad = False
                            
                            flat_st = norm_statements.view(N * 10, dim_in)
                            proj_flat_st = mnc_linear(flat_st)
                            proj_flat_st = (proj_flat_st + shift) / scale
                            proj_flat_st = torch.tanh(proj_flat_st)
                            statements = proj_flat_st.view(N, 10, dim_proj)
                            
                            proj_q = mnc_linear(norm_queries)
                            proj_q = (proj_q + shift) / scale
                            queries = torch.tanh(proj_q)
                            
                            dist_mean, dist_med, dist_p95 = compute_jl_distortion(
                                norm_statements.view(N * 10, dim_in),
                                proj_flat_st,
                                sample_size=min(5000, N * 10)
                            )
                        elif proj_method == "svd":
                            flat_st = norm_statements.view(N * 10, dim_in)
                            X_mean = flat_st.mean(dim=0, keepdim=True)
                            flat_st_centered = flat_st - X_mean
                            
                            _, _, V = torch.linalg.svd(flat_st_centered, full_matrices=False)
                            proj_matrix = V[:dim_proj].t()
                            
                            statements = torch.matmul(flat_st_centered, proj_matrix).view(N, 10, dim_proj)
                            queries = torch.matmul(norm_queries - X_mean, proj_matrix)
                            
                            dist_mean, dist_med, dist_p95 = compute_jl_distortion(
                                norm_statements.view(N * 10, dim_in),
                                statements.view(N * 10, dim_proj),
                                sample_size=min(5000, N * 10)
                            )
                        else:
                            statements = norm_statements.clone()
                            queries = norm_queries.clone()
                            dist_mean, dist_med, dist_p95 = 0.0, 0.0, 0.0
                            
                        svc_acc, lr_acc = run_linear_probe(statements, queries, labels, N)
                        for metric in ["L1", "L2", "Cosine"]:
                            metrics_res = run_metric_evaluation(statements, queries, labels, metric)
                            
                            row = {
                                "encoder": encoder_name,
                                "norm_mode": norm_mode,
                                "space": space_name,
                                "projection": proj_method,
                                "N": N,
                                "metric": metric,
                                "recall_proto": metrics_res["recall_proto"],
                                "recall_1nn": metrics_res["recall_1nn"],
                                "proto_advantage": metrics_res["proto_advantage"],
                                "mean_lcomp": metrics_res["mean_lcomp"],
                                "centroid_top1": metrics_res["centroid_topk"][1],
                                "centroid_top5": metrics_res["centroid_topk"][5],
                                "centroid_top10": metrics_res["centroid_topk"][10],
                                "centroid_top20": metrics_res["centroid_topk"][20],
                                "exemplar_top1": metrics_res["exemplar_topk"][1],
                                "exemplar_top5": metrics_res["exemplar_topk"][5],
                                "exemplar_top10": metrics_res["exemplar_topk"][10],
                                "exemplar_top20": metrics_res["exemplar_topk"][20],
                                "mean_proto_rank": metrics_res["mean_proto_rank"],
                                "median_proto_rank": metrics_res["median_proto_rank"],
                                "p90_proto_rank": metrics_res["p90_proto_rank"],
                                "mean_exemplar_rank": metrics_res["mean_exemplar_rank"],
                                "median_exemplar_rank": metrics_res["median_exemplar_rank"],
                                "p90_exemplar_rank": metrics_res["p90_exemplar_rank"],
                                "linear_probe_svc": svc_acc,
                                "linear_probe_lr": lr_acc,
                                "margin_p1": metrics_res["margin_percentiles"]["p1"],
                                "margin_p5": metrics_res["margin_percentiles"]["p5"],
                                "margin_p10": metrics_res["margin_percentiles"]["p10"],
                                "margin_p25": metrics_res["margin_percentiles"]["p25"],
                                "margin_p50": metrics_res["margin_percentiles"]["p50"],
                                "margin_p75": metrics_res["margin_percentiles"]["p75"],
                                "margin_p90": metrics_res["margin_percentiles"]["p90"],
                                "z_score": metrics_res["z_score"],
                                "distortion_mean": dist_mean,
                                "distortion_median": dist_med,
                                "distortion_p95": dist_p95
                            }
                            
                            with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=headers)
                                writer.writerow(row)
                                
                            if N == 800 and norm_mode == "l2_normalized" and space_name == "raw" and metric == "Cosine":
                                if encoder_name not in encoder_800_recalls:
                                    encoder_800_recalls[encoder_name] = metrics_res["recall_proto"]

    # --- SECOND PASS: Identify best alternative and extend both to N=1600 ---
    alternative_recalls = {k: v for k, v in encoder_800_recalls.items() if k != "all-MiniLM-L6-v2"}
    best_alternative = max(alternative_recalls, key=alternative_recalls.get) if alternative_recalls else None
    
    print(f"\n[+] Relational Cosine Recall at N=800:")
    for enc, rec in encoder_800_recalls.items():
        print(f"    * {enc}: {rec * 100.0:.2f}%")
    print(f"[+] Selected best alternative encoder: {best_alternative}")
    
    second_pass_encoders = ["all-MiniLM-L6-v2"]
    if best_alternative and best_alternative not in second_pass_encoders:
        second_pass_encoders.append(best_alternative)
        
    for encoder_name in second_pass_encoders:
        print(f"\n[+] Running Second Pass: {encoder_name} at N=1600...")
        pipeline = JournalPipeline(encoder_mapping[encoder_name])
        dim_in = pipeline.encoder.get_sentence_embedding_dimension()
        
        N = 1600
        facts = generate_facts(N)
        labels = torch.tensor([f["label"] for f in facts])
        
        is_e5 = "e5" in encoder_name.lower()
        with torch.no_grad():
            if is_e5:
                all_statements = [f"passage: {s}" for f in facts for s in f["statements"]]
                all_queries = [f"query: {f['query']}" for f in facts]
            else:
                all_statements = [s for f in facts for s in f["statements"]]
                all_queries = [f["query"] for f in facts]
            
            statements_flat = pipeline.encoder.encode(all_statements, batch_size=256, convert_to_tensor=True, device='cpu')
            raw_statements = statements_flat.view(N, 10, dim_in)
            raw_queries = pipeline.encoder.encode(all_queries, batch_size=256, convert_to_tensor=True, device='cpu')
            
        for norm_mode in ["raw", "l2_normalized"]:
            print(f"      Normalization mode: {norm_mode}")
            if norm_mode == "l2_normalized":
                norm_statements = raw_statements / (raw_statements.norm(p=2, dim=2, keepdim=True) + 1e-8)
                norm_queries = raw_queries / (raw_queries.norm(p=2, dim=1, keepdim=True) + 1e-8)
            else:
                norm_statements = raw_statements.clone()
                norm_queries = raw_queries.clone()
                
            spaces = [
                {"name": "raw", "dim": dim_in, "project": False},
                {"name": "proj_128d", "dim": 128, "project": True},
                {"name": "proj_256d", "dim": 256, "project": True}
            ]
            
            for space_cfg in spaces:
                space_name = space_cfg["name"]
                dim_proj = space_cfg["dim"]
                
                proj_methods = ["none"]
                if space_cfg["project"]:
                    proj_methods = ["random", "svd"]
                    
                for proj_method in proj_methods:
                    if proj_method == "random":
                        set_seed(42)
                        shift, scale = autocalibrate_scale_distances(dim_proj)
                        mnc_linear = MNCLinear(dim_in, dim_proj)
                        for param in mnc_linear.parameters():
                            param.requires_grad = False
                        
                        flat_st = norm_statements.view(N * 10, dim_in)
                        proj_flat_st = mnc_linear(flat_st)
                        proj_flat_st = (proj_flat_st + shift) / scale
                        proj_flat_st = torch.tanh(proj_flat_st)
                        statements = proj_flat_st.view(N, 10, dim_proj)
                        
                        proj_q = mnc_linear(norm_queries)
                        proj_q = (proj_q + shift) / scale
                        queries = torch.tanh(proj_q)
                        
                        dist_mean, dist_med, dist_p95 = compute_jl_distortion(
                            norm_statements.view(N * 10, dim_in),
                            proj_flat_st,
                            sample_size=min(5000, N * 10)
                        )
                    elif proj_method == "svd":
                        flat_st = norm_statements.view(N * 10, dim_in)
                        X_mean = flat_st.mean(dim=0, keepdim=True)
                        flat_st_centered = flat_st - X_mean
                        
                        _, _, V = torch.linalg.svd(flat_st_centered, full_matrices=False)
                        proj_matrix = V[:dim_proj].t()
                        
                        statements = torch.matmul(flat_st_centered, proj_matrix).view(N, 10, dim_proj)
                        queries = torch.matmul(norm_queries - X_mean, proj_matrix)
                        
                        dist_mean, dist_med, dist_p95 = compute_jl_distortion(
                            norm_statements.view(N * 10, dim_in),
                            statements.view(N * 10, dim_proj),
                            sample_size=min(5000, N * 10)
                        )
                    else:
                        statements = norm_statements.clone()
                        queries = norm_queries.clone()
                        dist_mean, dist_med, dist_p95 = 0.0, 0.0, 0.0
                        
                    svc_acc, lr_acc = run_linear_probe(statements, queries, labels, N)
                    for metric in ["L1", "L2", "Cosine"]:
                        metrics_res = run_metric_evaluation(statements, queries, labels, metric)
                        
                        row = {
                            "encoder": encoder_name,
                            "norm_mode": norm_mode,
                            "space": space_name,
                            "projection": proj_method,
                            "N": N,
                            "metric": metric,
                            "recall_proto": metrics_res["recall_proto"],
                            "recall_1nn": metrics_res["recall_1nn"],
                            "proto_advantage": metrics_res["proto_advantage"],
                            "mean_lcomp": metrics_res["mean_lcomp"],
                            "centroid_top1": metrics_res["centroid_topk"][1],
                            "centroid_top5": metrics_res["centroid_topk"][5],
                            "centroid_top10": metrics_res["centroid_topk"][10],
                            "centroid_top20": metrics_res["centroid_topk"][20],
                            "exemplar_top1": metrics_res["exemplar_topk"][1],
                            "exemplar_top5": metrics_res["exemplar_topk"][5],
                            "exemplar_top10": metrics_res["exemplar_topk"][10],
                            "exemplar_top20": metrics_res["exemplar_topk"][20],
                            "mean_proto_rank": metrics_res["mean_proto_rank"],
                            "median_proto_rank": metrics_res["median_proto_rank"],
                            "p90_proto_rank": metrics_res["p90_proto_rank"],
                            "mean_exemplar_rank": metrics_res["mean_exemplar_rank"],
                            "median_exemplar_rank": metrics_res["median_exemplar_rank"],
                            "p90_exemplar_rank": metrics_res["p90_exemplar_rank"],
                            "linear_probe_svc": svc_acc,
                            "linear_probe_lr": lr_acc,
                            "margin_p1": metrics_res["margin_percentiles"]["p1"],
                            "margin_p5": metrics_res["margin_percentiles"]["p5"],
                            "margin_p10": metrics_res["margin_percentiles"]["p10"],
                            "margin_p25": metrics_res["margin_percentiles"]["p25"],
                            "margin_p50": metrics_res["margin_percentiles"]["p50"],
                            "margin_p75": metrics_res["margin_percentiles"]["p75"],
                            "margin_p90": metrics_res["margin_percentiles"]["p90"],
                            "z_score": metrics_res["z_score"],
                            "distortion_mean": dist_mean,
                            "distortion_median": dist_med,
                            "distortion_p95": dist_p95
                        }
                        
                        with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=headers)
                            writer.writerow(row)
                            
    plot_results(csv_path, results_dir)
    generate_report(csv_path, results_dir)

def plot_results(csv_path, results_dir):
    raw_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw_data.append(r)
            
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    
    # 1. Recall vs N (Cosine, L2-normalized) for Raw Space
    ax = axes[0, 0]
    encoders = sorted(list(set([r["encoder"] for r in raw_data])))
    for enc in encoders:
        subset = [r for r in raw_data if r["encoder"] == enc and r["space"] == "raw" and r["norm_mode"] == "l2_normalized" and r["metric"] == "Cosine"]
        subset = sorted(subset, key=lambda x: int(x["N"]))
        N_vals = [int(x["N"]) for x in subset]
        recalls = [float(x["recall_proto"]) for x in subset]
        ax.plot(N_vals, recalls, 'o-', label=f"{enc} (Proto)")
    ax.set_title("Raw Cosine (L2-normalized) Recall vs. N")
    ax.set_xlabel("Fact Horizon (N)")
    ax.set_ylabel("Recall Accuracy")
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    
    # 2. Linear Probe vs Prototype Recall vs N (MiniLM, Normalized Cosine)
    ax = axes[0, 1]
    subset_minilm = [r for r in raw_data if r["encoder"] == "all-MiniLM-L6-v2" and r["space"] == "raw" and r["norm_mode"] == "l2_normalized" and r["metric"] == "Cosine"]
    subset_minilm = sorted(subset_minilm, key=lambda x: int(x["N"]))
    N_vals = [int(x["N"]) for x in subset_minilm]
    rec_proto = [float(x["recall_proto"]) for x in subset_minilm]
    rec_1nn = [float(x["recall_1nn"]) for x in subset_minilm]
    probe_svc = [float(x["linear_probe_svc"]) for x in subset_minilm]
    
    ax.plot(N_vals, rec_proto, 'o-', label="Prototype Retrieval")
    ax.plot(N_vals, rec_1nn, 's-', label="1-NN Retrieval")
    ax.plot(N_vals, probe_svc, '^-', label="Linear Separability Probe (SVC)")
    ax.set_title("MiniLM: Probe vs Retrieval (L2-norm, Cosine)")
    ax.set_xlabel("Fact Horizon (N)")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    
    # 3. Oracle Centroid vs Exemplar Top-K (MiniLM, N=800, L2-norm, Cosine)
    ax = axes[1, 0]
    minilm_800 = [r for r in raw_data if r["encoder"] == "all-MiniLM-L6-v2" and r["N"] == "800" and r["space"] == "raw" and r["norm_mode"] == "l2_normalized" and r["metric"] == "Cosine"][0]
    k_vals = [1, 5, 10, 20]
    c_topk = [float(minilm_800[f"centroid_top{k}"]) for k in k_vals]
    e_topk = [float(minilm_800[f"exemplar_top{k}"]) for k in k_vals]
    
    ax.plot(k_vals, c_topk, 'o-', label="Centroid Oracle")
    ax.plot(k_vals, e_topk, 's-', label="Exemplar Oracle")
    ax.set_title("MiniLM N=800: Oracle Centroid vs. Exemplar Top-K")
    ax.set_xlabel("K Neighbors")
    ax.set_ylabel("Oracle Recall")
    ax.set_xticks(k_vals)
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    
    # 4. Rank statistics of Correct Class (Raw Cosine, L2-normalized, N=800)
    ax = axes[1, 1]
    rank_subset = [r for r in raw_data if r["N"] == "800" and r["space"] == "raw" and r["norm_mode"] == "l2_normalized" and r["metric"] == "Cosine"]
    rank_subset = sorted(rank_subset, key=lambda x: x["encoder"])
    enc_labels = [x["encoder"][:15] + "..." for x in rank_subset]
    med_proto_ranks = [float(x["median_proto_rank"]) for x in rank_subset]
    med_ex_ranks = [float(x["median_exemplar_rank"]) for x in rank_subset]
    
    x = np.arange(len(enc_labels))
    width = 0.35
    ax.bar(x - width/2, med_proto_ranks, width, label="Prototype Median Rank")
    ax.bar(x + width/2, med_ex_ranks, width, label="Exemplar Median Rank")
    ax.set_title("N=800 Median Rank of Correct Class")
    ax.set_xticks(x)
    ax.set_xticklabels(enc_labels, rotation=15, ha='right')
    ax.set_ylabel("Median Rank (Lower is Better)")
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # 5. Projection Distortion vs Recall Loss Scatter Plot (MiniLM, N=800, L1 metric)
    ax = axes[2, 0]
    proj_subset = [r for r in raw_data if r["encoder"] == "all-MiniLM-L6-v2" and r["N"] == "800" and r["space"] != "raw" and r["metric"] == "L1"]
    raw_l1_rec = float([r for r in raw_data if r["encoder"] == "all-MiniLM-L6-v2" and r["N"] == "800" and r["space"] == "raw" and r["metric"] == "L1" and r["norm_mode"] == "raw"][0]["recall_proto"])
    
    for r in proj_subset:
        rec_loss = raw_l1_rec - float(r["recall_proto"])
        med_dist = float(r["distortion_median"])
        lbl = f"{r['space']} ({r['projection']})"
        ax.scatter(med_dist, rec_loss, s=100, label=lbl)
    
    ax.set_title("MiniLM N=800 L1: Median Distortion vs. Recall Loss")
    ax.set_xlabel("Median Distortion (Pairwise)")
    ax.set_ylabel("Recall Loss (Raw - Projected)")
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # 6. Margin CDF tail distribution (N=800, Raw space, Cosine, normalized)
    ax = axes[2, 1]
    pcts = [1, 5, 10, 25, 50, 75, 90]
    for enc in encoders:
        r_val = [r for r in raw_data if r["encoder"] == enc and r["N"] == "800" and r["space"] == "raw" and r["norm_mode"] == "l2_normalized" and r["metric"] == "Cosine"][0]
        margin_vals = [float(r_val[f"margin_p{p}"]) for p in pcts]
        ax.plot(margin_vals, pcts, 'o-', label=enc)
    ax.set_title("N=800 Cosine: Margin CDF Tail Distribution")
    ax.set_xlabel("Margin Value")
    ax.set_ylabel("Cumulative Percentile (%)")
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # 7. Prototype Advantage vs N (Raw Cosine, L2-normalized)
    ax = axes[3, 0]
    for enc in encoders:
        subset = [r for r in raw_data if r["encoder"] == enc and r["space"] == "raw" and r["norm_mode"] == "l2_normalized" and r["metric"] == "Cosine"]
        subset = sorted(subset, key=lambda x: int(x["N"]))
        N_vals = [int(x["N"]) for x in subset]
        advs = [float(x["proto_advantage"]) for x in subset]
        ax.plot(N_vals, advs, 'o-', label=enc)
    ax.set_title("Prototype Advantage (Prototype - 1-NN) vs. N")
    ax.set_xlabel("Fact Horizon (N)")
    ax.set_ylabel("Advantage Delta (Recall)")
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # 8. SVD vs. Random Projection recall (MiniLM, Cosine, Normalized)
    ax = axes[3, 1]
    svd_rand_subset = [r for r in raw_data if r["encoder"] == "all-MiniLM-L6-v2" and r["space"] != "raw" and r["metric"] == "Cosine" and r["norm_mode"] == "l2_normalized"]
    n_values = sorted(list(set([int(r["N"]) for r in svd_rand_subset])))
    
    for s_name in ["proj_128d", "proj_256d"]:
        for p_name in ["random", "svd"]:
            recalls = []
            for n in n_values:
                match = [r for r in svd_rand_subset if r["space"] == s_name and r["projection"] == p_name and int(r["N"]) == n]
                recalls.append(float(match[0]["recall_proto"]) if match else 0.0)
            ax.plot(n_values, recalls, 'o--', label=f"{s_name} ({p_name})")
            
    ax.set_title("MiniLM: Random vs. SVD Projection (Diagnostic Upper Bound)")
    ax.set_xlabel("Fact Horizon (N)")
    ax.set_ylabel("Recall Accuracy")
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "relational_manifold_audit.png")
    plt.savefig(plot_path, dpi=150)
    print(f"[*] Saved plot to: {plot_path}")

def generate_report(csv_path, results_dir):
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            data.append(r)
            
    md_path = os.path.join(results_dir, "relational_manifold_audit.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Experiment 3 Summary: Relational Manifold Audit & Oracles\n\n")
        f.write("## 🏆 Scientific Verdict\n\n")
        
        # 1. Main cross encoder table
        f.write("### 1. Cross-Encoder Relational Comparison (Raw Space, Cosine Metric, L2-normalized, N=800)\n\n")
        f.write("| Encoder | Recall (Prototype) | Recall (1-NN) | Prototype Advantage | Linear Separability Probe (SVC) | Oracle Centroid (Top-10) | Oracle Exemplar (Top-10) | Median Rank (Proto) | Median Rank (Exemplar) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        encoders = sorted(list(set([r["encoder"] for r in data])))
        for enc in encoders:
            match = [r for r in data if r["encoder"] == enc and r["N"] == "800" and r["space"] == "raw" and r["norm_mode"] == "l2_normalized" and r["metric"] == "Cosine"]
            if match:
                m = match[0]
                rec_p = float(m["recall_proto"]) * 100.0
                rec_1 = float(m["recall_1nn"]) * 100.0
                adv = float(m["proto_advantage"]) * 100.0
                svc = float(m["linear_probe_svc"]) * 100.0
                c_top = float(m["centroid_top10"]) * 100.0
                e_top = float(m["exemplar_top10"]) * 100.0
                m_p_r = float(m["median_proto_rank"])
                m_e_r = float(m["median_exemplar_rank"])
                f.write(f"| **{enc}** | {rec_p:.1f}% | {rec_1:.1f}% | {adv:+.1f}% | {svc:.1f}% | {c_top:.1f}% | {e_top:.1f}% | {m_p_r:.1f} | {m_e_r:.1f} |\n")
                
        # 2. Embedding normalization control table
        f.write("\n### 2. Embedding Normalization Impact (all-MiniLM-L6-v2, Raw Space, L1 Metric, N=800)\n\n")
        f.write("| Normalization | Recall (Prototype) | Recall (1-NN) | Prototype Advantage | Median Rank (Proto) | Median Rank (Exemplar) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for n_mode in ["raw", "l2_normalized"]:
            match = [r for r in data if r["encoder"] == "all-MiniLM-L6-v2" and r["N"] == "800" and r["space"] == "raw" and r["norm_mode"] == n_mode and r["metric"] == "L1"]
            if match:
                m = match[0]
                rec_p = float(m["recall_proto"]) * 100.0
                rec_1 = float(m["recall_1nn"]) * 100.0
                adv = float(m["proto_advantage"]) * 100.0
                m_p_r = float(m["median_proto_rank"])
                m_e_r = float(m["median_exemplar_rank"])
                f.write(f"| {n_mode} | {rec_p:.1f}% | {rec_1:.1f}% | {adv:+.1f}% | {m_p_r:.1f} | {m_e_r:.1f} |\n")
                
        # 3. Projection distortion table
        f.write("\n### 3. Projection Distortion vs. Data-Dependent SVD (Diagnostic Upper Bound) (all-MiniLM-L6-v2, N=800, L1 Metric)\n\n")
        f.write("| Space | Projection | Recall (Prototype) | Recall Loss | Median Distortion (ε) | 95th Percentile Distortion |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        
        raw_l1 = [r for r in data if r["encoder"] == "all-MiniLM-L6-v2" and r["N"] == "800" and r["space"] == "raw" and r["metric"] == "L1" and r["norm_mode"] == "raw"][0]
        raw_rec = float(raw_l1["recall_proto"])
        
        for s_name in ["proj_128d", "proj_256d"]:
            for p_name in ["random", "svd"]:
                match = [r for r in data if r["encoder"] == "all-MiniLM-L6-v2" and r["N"] == "800" and r["space"] == s_name and r["projection"] == p_name and r["metric"] == "L1"]
                if match:
                    m = match[0]
                    rec_p = float(m["recall_proto"]) * 100.0
                    loss = (raw_rec - float(m["recall_proto"])) * 100.0
                    med_d = float(m["distortion_median"])
                    p95_d = float(m["distortion_p95"])
                    f.write(f"| {s_name} | {p_name} | {rec_p:.1f}% | {loss:.1f}% | {med_d:.4f} | {p95_d:.4f} |\n")
                    
        # 4. Margin percentiles
        f.write("\n### 4. Margin CDF Tail Distribution (N=800, Raw Space, Cosine Metric, L2-normalized)\n\n")
        f.write("| Encoder | 1st Percentile | 5th Percentile | 10th Percentile | 25th Percentile | 50th Percentile (Median) | 90th Percentile |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for enc in encoders:
            match = [r for r in data if r["encoder"] == enc and r["N"] == "800" and r["space"] == "raw" and r["norm_mode"] == "l2_normalized" and r["metric"] == "Cosine"]
            if match:
                m = match[0]
                p1 = float(m["margin_p1"])
                p5 = float(m["margin_p5"])
                p10 = float(m["margin_p10"])
                p25 = float(m["margin_p25"])
                p50 = float(m["margin_p50"])
                p90 = float(m["margin_p90"])
                f.write(f"| **{enc}** | {p1:.4f} | {p5:.4f} | {p10:.4f} | {p25:.4f} | {p50:.4f} | {p90:.4f} |\n")
                
        f.write("\n\n## 🔍 Interpretation & Recommendations\n")
        f.write("Use these findings to update the project roadmap:\n")
        f.write("1. **Is the manifold structurally weak or is ranking failing?** Compare the Oracle Centroid (Top-10) and Exemplar (Top-10) with the Prototype recall. If Oracle is > 80% while Prototype is < 50%, the information is present and metrically accessible, but distance ranking requires a non-linear or multi-prototype scheme.\n")
        f.write("2. **Is there an encoder ceiling?** Compare maximum recalls across encoders at N=800. If all encoders plateau at similar levels (< 60%), the relational collision dataset exposes a general geometric packing limit of sentence transformers.\n")
        f.write("3. **Does L2-Normalization fix metric degradation?** Compare raw vs L2-normalized recall. If E5/MPNet/BGE recall jumps significantly under L2-normalization, cosine similarity on normalized vectors must be enforced throughout the MNC layers.\n")
        
    print(f"[*] Saved markdown report to: {md_path}")

if __name__ == "__main__":
    main()
