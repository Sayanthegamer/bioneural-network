import sys
import os
import torch
import torch.nn as nn
import numpy as np
import random
from sentence_transformers import SentenceTransformer

# Add mnc_project to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mnc_project'))
from pipeline import JournalPipeline

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    print("========================================================================")
    print("  RAW UNCOMPRESSED MINILM RETRIEVAL BASELINE (384D Space)")
    print("========================================================================")
    
    pipeline = JournalPipeline()
    
    # We will test N facts
    N_list = [5, 10, 20, 50, 100, 200, 400, 800, 1600]
    
    print("\n| N Facts | L1 Recall | L2 Recall | Cosine Recall | Mean L1 Margin | Sep Ratio (L1) |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for N in N_list:
        set_seed(42)
        facts = generate_facts(N)
        
        # Embed all statements and queries
        with torch.no_grad():
            statements = [f["statement"] for f in facts]
            queries = [f["query"] for f in facts]
            
            # Generate raw embeddings on CPU
            # Shape: [N, 384]
            s_embs = pipeline.encoder.encode(statements, convert_to_tensor=True, device='cpu')
            q_embs = pipeline.encoder.encode(queries, convert_to_tensor=True, device='cpu')
            
            # 1. L1 Distance evaluation
            # s_embs: [N, 384] -> [1, N, 384]
            # q_embs: [N, 384] -> [N, 1, 384]
            diff = q_embs.unsqueeze(1) - s_embs.unsqueeze(0) # [N, N, 384]
            l1_dists = diff.abs().sum(dim=2) # [N, N]
            
            l1_preds = l1_dists.argmin(dim=1).tolist()
            l1_correct = sum(1 for idx in range(N) if l1_preds[idx] == facts[idx]["label"])
            l1_recall = l1_correct / float(N)
            
            # Telemetry for L1
            radii_l1 = l1_dists[range(N), range(N)]
            mean_radius_l1 = radii_l1.mean().item()
            
            mask = torch.eye(N, device=l1_dists.device) * 1e9
            nearest_others_l1 = (l1_dists + mask).min(dim=1).values
            mean_nearest_other_l1 = nearest_others_l1.mean().item()
            
            sep_ratio_l1 = mean_nearest_other_l1 / (mean_radius_l1 + 1e-8)
            mean_margin_l1 = (nearest_others_l1 - radii_l1).mean().item()
            
            # 2. L2 Distance evaluation
            l2_dists = torch.norm(diff, p=2, dim=2) # [N, N]
            l2_preds = l2_dists.argmin(dim=1).tolist()
            l2_correct = sum(1 for idx in range(N) if l2_preds[idx] == facts[idx]["label"])
            l2_recall = l2_correct / float(N)
            
            # 3. Cosine Similarity evaluation
            # Normalize embeddings to unit sphere
            s_embs_norm = s_embs / (s_embs.norm(p=2, dim=1, keepdim=True) + 1e-8)
            q_embs_norm = q_embs / (q_embs.norm(p=2, dim=1, keepdim=True) + 1e-8)
            cosine_sims = torch.mm(q_embs_norm, s_embs_norm.t()) # [N, N]
            
            cosine_preds = cosine_sims.argmax(dim=1).tolist()
            cosine_correct = sum(1 for idx in range(N) if cosine_preds[idx] == facts[idx]["label"])
            cosine_recall = cosine_correct / float(N)
            
            print(f"| {N:7d} | {l1_recall*100:8.2f}% | {l2_recall*100:8.2f}% | {cosine_recall*100:12.2f}% | {mean_margin_l1:13.4f} | {sep_ratio_l1:13.4f} |")

if __name__ == "__main__":
    main()
