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

# Autocalibrate globally
SHIFT0, SCALE0 = autocalibrate_scale_distances(384)

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

def run_prototypical_capacity_wall(num_facts, seeds, csv_path):
    print(f"\n====================================================")
    print(f"  Prototypical Capacity Wall Sweep: N = {num_facts}")
    print(f"====================================================")
    
    pipeline = JournalPipeline()
    facts = generate_facts(num_facts)
    
    print(f"[*] Pre-embedding {num_facts} facts and queries...")
    embedded_statements = [pipeline.embed_sentence(f["statement"]) for f in facts]
    embedded_queries = [pipeline.embed_sentence(f["query"]) for f in facts]
    print("[*] Embedding complete.")
    
    results = []
    
    for seed in seeds:
        t_start = time.time()
        set_seed(seed)
        
        # 1. Instantiate backbone layer and autocalibrate scale
        backbone_linear = MNCLinear(384, 32)
        # Freeze weights of the projection layers right after initialization (no representation drift)
        for param in backbone_linear.parameters():
            param.requires_grad = False
            
        backbone = nn.Sequential(
            backbone_linear,
            ScaleDistances(SHIFT0, SCALE0),
            nn.Tanh()
        )
        backbone.eval()
        
        # 2. Instantiate Prototypical Network
        model = MNCPrototypicalNetwork(backbone, bottleneck_dim=32, num_classes=num_facts)
        model.eval()
        
        # 3. Sequential Ingestion: Add facts one by one
        statement_projections = []
        fact_labels = [f["label"] for f in facts]
        
        with torch.no_grad():
            for idx in range(num_facts):
                statement_vec = embedded_statements[idx]
                label = facts[idx]["label"]
                
                # Add fact to model (computes running average prototype)
                model.add_fact(statement_vec, label)
                
                # Keep raw statement projection locally for evaluation-only KNN
                z_proj = model.backbone(statement_vec).squeeze(0)
                statement_projections.append(z_proj)
                
            statement_projections = torch.stack(statement_projections)
            query_projections = torch.stack([model.backbone(q_vec).squeeze(0) for q_vec in embedded_queries])
            
            # 4. Evaluate: Prototype Averaging
            proto_correct = 0
            proto_predictions = []
            for idx in range(num_facts):
                logits = model(embedded_queries[idx])
                pred = logits.argmax(dim=-1).item()
                proto_predictions.append(pred)
                if pred == facts[idx]["label"]:
                    proto_correct += 1
            proto_acc = proto_correct / float(num_facts)
            
            # 5. Evaluate: 1-NN Exemplar classification (decoupled)
            knn1_correct = 0
            for idx in range(num_facts):
                pred1 = classify_knn(query_projections[idx], statement_projections, fact_labels, k=1)
                if pred1 == facts[idx]["label"]:
                    knn1_correct += 1
            knn1_acc = knn1_correct / float(num_facts)
            
            # 6. Evaluate: 3-NN Exemplar classification (decoupled)
            knn3_correct = 0
            for idx in range(num_facts):
                pred3 = classify_knn(query_projections[idx], statement_projections, fact_labels, k=3)
                if pred3 == facts[idx]["label"]:
                    knn3_correct += 1
            knn3_acc = knn3_correct / float(num_facts)
            
            # 7. Random-Label Permutation Sanity Check (for small N)
            permuted_acc = 0.0
            if num_facts <= 50:
                permuted_labels = fact_labels.copy()
                random.shuffle(permuted_labels)
                p_correct = 0
                for idx in range(num_facts):
                    pred_proto = model(embedded_queries[idx]).argmax(dim=-1).item()
                    if permuted_labels[pred_proto] == facts[idx]["label"]:
                        p_correct += 1
                permuted_acc = p_correct / float(num_facts)
                
            # 8. Geometry Telemetry: Margins & Radii
            same_class_radii = []
            inter_class_distances = []
            prototype_margins = []
            
            for idx in range(num_facts):
                correct_proto = model.prototypes[idx]
                q_z = query_projections[idx]
                
                # Same-Class Radius (L1 distance to own prototype)
                radius = (q_z - correct_proto).abs().sum().item()
                same_class_radii.append(radius)
                
                # Other distances
                other_dists = []
                for other_idx in range(num_facts):
                    if other_idx != idx:
                        other_proto = model.prototypes[other_idx]
                        dist = (q_z - other_proto).abs().sum().item()
                        other_dists.append(dist)
                
                nearest_other = min(other_dists) if len(other_dists) > 0 else 0.0
                inter_class_distances.append(nearest_other)
                prototype_margins.append(nearest_other - radius)
                
            mean_radius = np.mean(same_class_radii)
            mean_nearest_other = np.mean(inter_class_distances)
            mean_margin = np.mean(prototype_margins)
            
            # 9. Confusion Matrix Analysis (for key values N = 50, 200, 800)
            same_concept_errors = 0
            total_errors = 0
            if num_facts in [50, 200, 800] and seed == 0:
                print(f"\n[CONFUSION ANALYSIS] Detailed logs for Seed 0 (N = {num_facts}):")
                for idx in range(num_facts):
                    pred = proto_predictions[idx]
                    label = facts[idx]["label"]
                    if pred != label:
                        total_errors += 1
                        pred_template = facts[pred]["template_type"]
                        true_template = facts[idx]["template_type"]
                        if pred_template == true_template:
                            same_concept_errors += 1
                
                if total_errors > 0:
                    same_concept_pct = (same_concept_errors / total_errors) * 100.0
                    print(f"  Total Errors: {total_errors} | Same-Concept Errors: {same_concept_errors} ({same_concept_pct:.2f}%)")
                else:
                    print("  Perfect classification: 0 errors.")
                    
            runtime = time.time() - t_start
            
            res_row = {
                "seed": seed,
                "num_facts": num_facts,
                "prototype_acc": proto_acc,
                "knn1_acc": knn1_acc,
                "knn3_acc": knn3_acc,
                "permuted_acc": permuted_acc,
                "mean_radius": mean_radius,
                "mean_nearest_other": mean_nearest_other,
                "mean_margin": mean_margin,
                "runtime_sec": runtime
            }
            results.append(res_row)
            
            # Formatted log printing
            perm_str = f"| Permuted: {permuted_acc*100:5.1f}%" if num_facts <= 50 else ""
            print(f"  Seed {seed:3} | Proto: {proto_acc*100:5.1f}% | 1-NN: {knn1_acc*100:5.1f}% | Margin: {mean_margin:.4f} {perm_str} | Runtime: {runtime:.2f}s")
            
    # Append results to CSV file
    file_exists = os.path.exists(csv_path)
    headers = [
        "seed", "num_facts", "prototype_acc", "knn1_acc", "knn3_acc", 
        "permuted_acc", "mean_radius", "mean_nearest_other", "mean_margin", "runtime_sec"
    ]
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Calculate stats
    proto_accs = [r["prototype_acc"] for r in results]
    knn1_accs = [r["knn1_acc"] for r in results]
    knn3_accs = [r["knn3_acc"] for r in results]
    margins = [r["mean_margin"] for r in results]
    radii = [r["mean_radius"] for r in results]
    nearest_others = [r["mean_nearest_other"] for r in results]
    
    print(f"\n--> [SUMMARY] N = {num_facts}:")
    print(f"    Prototype Accuracy : {np.mean(proto_accs)*100:6.2f}% +/- {np.std(proto_accs)*100:5.2f}%")
    print(f"    1-NN Accuracy       : {np.mean(knn1_accs)*100:6.2f}% +/- {np.std(knn1_accs)*100:5.2f}%")
    print(f"    3-NN Accuracy       : {np.mean(knn3_accs)*100:6.2f}% +/- {np.std(knn3_accs)*100:5.2f}%")
    print(f"    Mean L1 Radius     : {np.mean(radii):.6f}")
    print(f"    Mean L1 Margin     : {np.mean(margins):.6f}")
    
    return np.mean(proto_accs), np.std(proto_accs), np.mean(knn1_accs), np.mean(margins)

if __name__ == "__main__":
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "prototypical_capacity.csv")
    
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    capacity_steps = [5, 10, 20, 50, 100, 200, 400, 800]
    
    summary_stats = []
    for step in capacity_steps:
        p_acc, p_std, k1_acc, margin = run_prototypical_capacity_wall(step, seeds, csv_path)
        summary_stats.append((step, p_acc, p_std, k1_acc, margin))
        
    print("\n====================================================================================================")
    print("  FINAL PROTOTYPICAL CAPACITY SWEEP SUMMARY")
    print("====================================================================================================")
    print(f"{'N Facts':<10} | {'Prototype Recall':<20} | {'1-NN Recall':<15} | {'Mean L1 Margin':<15}")
    print("-" * 70)
    for step, p_acc, p_std, k1_acc, margin in summary_stats:
        print(f"{step:<10} | {p_acc*100:5.2f}% +/- {p_std*100:5.2f}% | {k1_acc*100:5.2f}% | {margin:.4f}")
    print("====================================================================================================")
