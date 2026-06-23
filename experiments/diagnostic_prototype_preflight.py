import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random

# Add mnc_project to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mnc_project'))

from mnc.layers import MNCLinear
from mnc.memory import MESUEngine
from pipeline import JournalPipeline

FACTS = [
    {"statement": "The blue folder is in the third drawer.", "query": "Where is the blue folder kept?", "label": 0},
    {"statement": "The access code for the main server is 7734.", "query": "What's the main server's access code?", "label": 1},
    {"statement": "Sarah's strategy meeting was moved to Tuesday at 3 PM.", "query": "When did Sarah's strategy meeting get rescheduled to?", "label": 2},
    {"statement": "The backup generator requires unleaded fuel.", "query": "What kind of fuel does the backup generator need?", "label": 3},
    {"statement": "Sector 4 security cameras undergo maintenance at midnight.", "query": "When does Sector 4 camera maintenance happen?", "label": 4}
]

INTERFERENCE = [
    {"text": "The red folder is resting on the top desk.", "query": "Where is the red folder?", "label": 5},
    {"text": "The access code for the guest wifi is 9912.", "query": "What is the guest wifi access code?", "label": 6},
    {"text": "Someone left a coffee mug in the breakroom.", "query": "Where was the coffee mug left?", "label": 7},
    {"text": "It is raining heavily outside today.", "query": "What is the weather like today?", "label": 8}
]

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
SHIFT1, SCALE1 = autocalibrate_scale_distances(32)

def make_mnc_model():
    model = nn.Sequential(
        MNCLinear(384, 32),
        ScaleDistances(SHIFT0, SCALE0),
        nn.Tanh(),
        MNCLinear(32, 10),
        ScaleDistances(SHIFT1, SCALE1)
    )
    return model

def run_preflight_diagnostics():
    print("====================================================================================================")
    # Changed wording to scientifically test instead of assume
    print("  PRE-FLIGHT DIAGNOSTIC: Testing Representation vs. Readout vs. Prototype Crowding")
    print("====================================================================================================")
    
    pipeline = JournalPipeline()
    seeds = list(range(10))
    
    # Storage for aggregate statistics across seeds
    linear_recalls = []
    proto_recalls = []
    knn1_recalls = []
    knn3_recalls = []
    knn5_recalls = []
    
    # Permuted-label check statistics
    permuted_proto_recalls = []
    permuted_knn1_recalls = []
    
    # Telemetry geometry tracking
    all_same_class_radii = []
    all_inter_class_distances = []
    all_prototype_margins = []
    
    # Tensors for raw file saving (we will export the data from seed 0 as a representative case)
    export_embeddings = []
    export_labels = []
    
    for seed in seeds:
        set_seed(seed)
        model = make_mnc_model()
        engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=0.02, conductance_mode='negative', u2_enabled=True)
        
        # Build the exact Day 5 fact stream
        stream = []
        for f in FACTS:
            stream.append({"text": f["statement"], "label": f["label"], "is_fact": True})
        for dist in INTERFERENCE:
            stream.append({"text": dist["text"], "label": dist["label"], "is_fact": False})
            
        # 1. Train on the sequential facts (Days 1-5) and distractors (Days 6-9)
        model.train()
        for day_idx, item in enumerate(stream):
            vec = pipeline.embed_sentence(item["text"])
            num_steps = 15 if item["is_fact"] else 3
            
            for step_idx in range(num_steps):
                noise = torch.randn_like(vec) * 0.05
                noisy_vec = vec + noise
                logits = model(noisy_vec)
                
                # Margin Contrastive Loss
                loss = 0.0
                margin = 1.0
                for idx_class in range(10):
                    if idx_class != item["label"]:
                        loss += torch.clamp(logits[0, idx_class] - logits[0, item["label"]] + margin, min=0.0)
                
                loss.backward()
                engine.step(loss.item())
                engine.zero_grad()
                
        # 2. Freeze the representation backbone (MNC layer 0 + ScaleDistances + Tanh)
        model.eval()
        for param in model[0].parameters():
            param.requires_grad = False
            
        # 3. Export embeddings of statements and queries
        with torch.no_grad():
            statement_embeddings = [] # [5, 32]
            query_embeddings = [] # [5, 32]
            
            for f in FACTS:
                # Statement projection (acts as source for prototype/exemplars)
                s_vec = pipeline.embed_sentence(f["statement"])
                s_proj = model[0:3](s_vec).squeeze(0) # [32]
                statement_embeddings.append(s_proj)
                
                # Query projection (acts as search probe)
                q_vec = pipeline.embed_sentence(f["query"])
                q_proj = model[0:3](q_vec).squeeze(0) # [32]
                query_embeddings.append(q_proj)
                
            statement_embeddings = torch.stack(statement_embeddings)
            query_embeddings = torch.stack(query_embeddings)
            
            # Export raw embeddings for seed 0 to file
            if seed == 0:
                export_embeddings = statement_embeddings.clone()
                export_labels = torch.tensor([f["label"] for f in FACTS])
                
            # 4. Compute Prototype Centers (Averages)
            # Since there is one statement per fact, prototype is the statement embedding
            prototypes = statement_embeddings.clone() # [5, 32]
            
            # 5. Evaluate: Linear Classifier Readout
            linear_correct = 0
            for idx, f in enumerate(FACTS):
                q_vec = pipeline.embed_sentence(f["query"])
                logits = model(q_vec)
                pred = logits.argmax(dim=-1).item()
                if pred == f["label"]:
                    linear_correct += 1
            linear_recalls.append(linear_correct / 5.0)
            
            # Helper function: L1 classification
            def classify_l1(query_z, target_tensors):
                # query_z: [32], target_tensors: [N, 32]
                diff = target_tensors - query_z.unsqueeze(0)
                dists = diff.abs().sum(dim=1)
                return dists.argmin().item() # Returns index of closest target
                
            # Helper function: k-NN classification
            def classify_knn(query_z, exemplar_tensors, labels, k=1):
                diff = exemplar_tensors - query_z.unsqueeze(0)
                dists = diff.abs().sum(dim=1)
                topk_indices = dists.argsort()[:k]
                topk_labels = [labels[idx] for idx in topk_indices.tolist()]
                # Majority vote
                return max(set(topk_labels), key=topk_labels.count)
                
            # 6. Evaluate: Prototype Readout
            proto_correct = 0
            for idx, f in enumerate(FACTS):
                pred = classify_l1(query_embeddings[idx], prototypes)
                if pred == f["label"]:
                    proto_correct += 1
            proto_recalls.append(proto_correct / 5.0)
            
            # 7. Evaluate: 1-NN, 3-NN, 5-NN
            knn1_correct = 0
            knn3_correct = 0
            knn5_correct = 0
            fact_labels = [f["label"] for f in FACTS]
            
            for idx, f in enumerate(FACTS):
                pred1 = classify_knn(query_embeddings[idx], statement_embeddings, fact_labels, k=1)
                pred3 = classify_knn(query_embeddings[idx], statement_embeddings, fact_labels, k=3)
                pred5 = classify_knn(query_embeddings[idx], statement_embeddings, fact_labels, k=5)
                
                if pred1 == f["label"]: knn1_correct += 1
                if pred3 == f["label"]: knn3_correct += 1
                if pred5 == f["label"]: knn5_correct += 1
                
            knn1_recalls.append(knn1_correct / 5.0)
            knn3_recalls.append(knn3_correct / 5.0)
            knn5_recalls.append(knn5_correct / 5.0)
            
            # 8. Random-Label Permutation Sanity Check
            permuted_labels = fact_labels.copy()
            random.shuffle(permuted_labels)
            
            p_proto_correct = 0
            p_knn1_correct = 0
            for idx, f in enumerate(FACTS):
                pred_proto = classify_l1(query_embeddings[idx], prototypes)
                # Map prediction to permuted labels
                if permuted_labels[pred_proto] == f["label"]:
                    p_proto_correct += 1
                    
                pred_knn = classify_knn(query_embeddings[idx], statement_embeddings, permuted_labels, k=1)
                if pred_knn == f["label"]:
                    p_knn1_correct += 1
                    
            permuted_proto_recalls.append(p_proto_correct / 5.0)
            permuted_knn1_recalls.append(p_knn1_correct / 5.0)
            
            # 9. Geometry Telemetry: Margin, Radius, Nearest Other
            same_class_radii = []
            inter_class_distances = []
            prototype_margins = []
            
            for idx in range(len(FACTS)):
                # same class radius: L1 distance between query and correct prototype
                correct_proto = prototypes[idx]
                q_z = query_embeddings[idx]
                radius = (q_z - correct_proto).abs().sum().item()
                same_class_radii.append(radius)
                
                # other distances
                other_dists = []
                for other_idx in range(len(FACTS)):
                    if other_idx != idx:
                        other_proto = prototypes[other_idx]
                        dist = (q_z - other_proto).abs().sum().item()
                        other_dists.append(dist)
                        
                nearest_other = min(other_dists)
                inter_class_distances.append(nearest_other)
                prototype_margins.append(nearest_other - radius)
                
            all_same_class_radii.append(np.mean(same_class_radii))
            all_inter_class_distances.append(np.mean(inter_class_distances))
            all_prototype_margins.append(np.mean(prototype_margins))

    # Save raw tensors for seed 0 to results folder
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    torch.save(export_embeddings, os.path.join(results_dir, "embeddings.pt"))
    torch.save(export_labels, os.path.join(results_dir, "labels.pt"))
    print(f"[*] Exported raw embeddings and labels to: {results_dir}")

    # Print comparative evaluation table
    print("\n====================================================================================================")
    print("  COMPARATIVE EVALUATION SUMMARY (Averaged over 10 Seeds)")
    print("====================================================================================================")
    print(f"  Standard Linear Head Recall: {np.mean(linear_recalls)*100:6.2f}% +/- {np.std(linear_recalls)*100:5.2f}%")
    print(f"  Prototype Average Recall:    {np.mean(proto_recalls)*100:6.2f}% +/- {np.std(proto_recalls)*100:5.2f}%")
    print(f"  1-NN Exemplar Recall:        {np.mean(knn1_recalls)*100:6.2f}% +/- {np.std(knn1_recalls)*100:5.2f}%")
    print(f"  3-NN Exemplar Recall:        {np.mean(knn3_recalls)*100:6.2f}% +/- {np.std(knn3_recalls)*100:5.2f}%")
    print(f"  5-NN Exemplar Recall:        {np.mean(knn5_recalls)*100:6.2f}% +/- {np.std(knn5_recalls)*100:5.2f}%")
    
    print("\n====================================================================================================")
    print("  RANDOM-LABEL SANITY CHECKS (Should be near chance ~20%)")
    print("====================================================================================================")
    print(f"  Permuted-Label Prototype Recall: {np.mean(permuted_proto_recalls)*100:6.2f}% +/- {np.std(permuted_proto_recalls)*100:5.2f}%")
    print(f"  Permuted-Label 1-NN Recall:      {np.mean(permuted_knn1_recalls)*100:6.2f}% +/- {np.std(permuted_knn1_recalls)*100:5.2f}%")

    print("\n====================================================================================================")
    print("  GEOMETRY TELEMETRY (L1 Distance Space)")
    print("====================================================================================================")
    print(f"  Mean Same-Class Radius:       {np.mean(all_same_class_radii):.6f}")
    print(f"  Mean Nearest-Other Distance:  {np.mean(all_inter_class_distances):.6f}")
    print(f"  Mean Prototype Margin:        {np.mean(all_prototype_margins):.6f}")
    print("====================================================================================================")

if __name__ == "__main__":
    run_preflight_diagnostics()
