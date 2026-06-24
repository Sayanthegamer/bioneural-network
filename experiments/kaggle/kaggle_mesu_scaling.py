# ==============================================================================
# SELF-CONTAINED MESU WIDTH-SCALING & U2-ABLATION EXPERIMENTS FOR KAGGLE
# ==============================================================================
# This script is fully self-contained. You can paste it directly into a single
# Kaggle Notebook cell. It automatically handles package installation, embeds 
# the custom MNC layers/MESU engine, and executes the sweeps on Kaggle's GPU/CPU.
# ==============================================================================

import subprocess
import sys

# Ensure required packages are installed
try:
    import sentence_transformers
except ImportError:
    print("[*] Installing sentence-transformers package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])

import os
import csv
import time
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Set to True if GPU kernel overhead makes GPU execution slower than CPU.
# For tiny networks (batch size 1), CPU is often faster due to zero kernel launch latency.
FORCE_CPU = False

# ==============================================================================
# 1. CORE MESU ARCHITECTURE & MNC LAYERS
# ==============================================================================

class MNCAdderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        x_exp = x.unsqueeze(1)  # [Batch, 1, In_Features]
        w_exp = w.unsqueeze(0)  # [1, Out_Features, In_Features]
        diff = x_exp - w_exp    
        out = -torch.abs(diff).sum(dim=2)  # Shape: [Batch, Out_Features]
        ctx.save_for_backward(diff)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        diff, = ctx.saved_tensors
        grad_output_exp = grad_output.unsqueeze(2)
        grad_w_diff = diff  # (X - W)
        grad_w = (grad_output_exp * grad_w_diff).sum(dim=0)
        grad_x_diff = torch.clamp(diff, min=-1.0, max=1.0)
        grad_x = -(grad_output_exp * grad_x_diff).sum(dim=1)
        return grad_x, grad_w

def mnc_adder(x, w):
    return MNCAdderFunction.apply(x, w)

class MNCLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.W, mean=0.0, std=1.0)
        with torch.no_grad():
            self.W.data = self.W.data / (self.W.data.norm(p=2, dim=1, keepdim=True) + 1e-8)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        out = mnc_adder(x, self.W)
        return out + self.bias

class MESUEngine:
    def __init__(self, model, lr=2.0, sigma_prior=0.1, alpha_decay=0.01, u2_enabled=True):
        self.model = model
        self.lr = lr
        self.sigma_prior = sigma_prior
        self.alpha_decay = alpha_decay
        self.u2_enabled = u2_enabled
        self.variances = {}
        self.cascade_states = {}
        self._initialize_memory_states()

    def _initialize_memory_states(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.variances[name] = torch.full_like(param.data, self.sigma_prior ** 2)
                self.cascade_states[name + "_u1"] = torch.clone(param.data)
                self.cascade_states[name + "_u2"] = torch.clone(param.data)

    @torch.no_grad()
    def step(self, current_loss=None):
        g = 0.05
        if current_loss is not None:
            g = 0.1 * torch.sigmoid(-torch.tensor(current_loss)).item()

        for name, param in self.model.named_parameters():
            if param.grad is None: continue
            raw_grad = param.grad.data
            
            grad_norm = raw_grad.norm(2)
            if grad_norm > 1e-5:
                scale = (param.numel() ** 0.5) / (grad_norm + 1e-4)
                raw_grad = raw_grad * scale
                
            var = self.variances[name]
            effective_lr = self.lr * var 
            param.data.sub_(effective_lr * raw_grad)
            
            u1_key = name + "_u1"
            u2_key = name + "_u2"
            if u1_key in self.cascade_states and u2_key in self.cascade_states:
                u1 = self.cascade_states[u1_key]
                u2 = self.cascade_states[u2_key]
                u1.add_(g * (param.data - u1))
                u2.add_(0.1 * g * (u1 - u2))
                
                if self.u2_enabled:
                    confidence = 1.0 - (var / (self.sigma_prior ** 2))
                    confidence = torch.clamp(confidence, min=0.0, max=1.0)
                    param.data.add_(confidence * g * (u2 - param.data))
                
            if "W" in name and param.data.dim() == 2:
                param.data.copy_(param.data / (param.data.norm(p=2, dim=1, keepdim=True) + 1e-8))
                if u1_key in self.cascade_states:
                    self.cascade_states[u1_key].copy_(self.cascade_states[u1_key] / (self.cascade_states[u1_key].norm(p=2, dim=1, keepdim=True) + 1e-8))
                if u2_key in self.cascade_states:
                    self.cascade_states[u2_key].copy_(self.cascade_states[u2_key] / (self.cascade_states[u2_key].norm(p=2, dim=1, keepdim=True) + 1e-8))
            
            var.sub_(var * torch.clamp(raw_grad.abs() * 0.2, max=0.25))
            var.add_(self.alpha_decay * ((self.sigma_prior**2) - var))
            var.clamp_(min=1e-4, max=self.sigma_prior ** 2)

    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

# ==============================================================================
# 2. EXPERIMENT PIPELINE FUNCTIONS
# ==============================================================================

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

def evaluate_fact(model, fact, embedded_query, device):
    model.eval()
    with torch.no_grad():
        logits = model(embedded_query)
        pred = logits.argmax(dim=-1).item()
    return 1.0 if pred == fact["label"] else 0.0

def run_width_evaluation(width, num_facts, seeds, facts, embedded_statements, embedded_queries, shift_w, scale_w, shift_384, scale_384, u2_enabled, device):
    results = []
    for seed in seeds:
        t_start = time.time()
        set_seed(seed)
        
        # Build model on target device
        model = nn.Sequential(
            MNCLinear(384, width),
            ScaleDistances(shift_384, scale_384),
            nn.Tanh(),
            MNCLinear(width, num_facts),
            ScaleDistances(shift_w, scale_w)
        ).to(device)
        
        W0_layer0 = torch.clone(model[0].W.data)
        W0_layer3 = torch.clone(model[3].W.data)
        
        engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=0.02, u2_enabled=u2_enabled)
        loss_fn = nn.CrossEntropyLoss()
        
        initial_recalls = []
        
        # Train sequentially
        for idx in range(num_facts):
            statement_vec = embedded_statements[idx]  # Already on device
            target = torch.tensor([idx], device=device)  # Created directly on device
            
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
            acc_init = evaluate_fact(model, facts[idx], embedded_queries[idx], device)
            initial_recalls.append(acc_init)
            
        # Batched final recall evaluation
        model.eval()
        with torch.no_grad():
            qvecs = torch.cat(embedded_queries, dim=0)  # Already on device
            logits = model(qvecs)
            preds = logits.argmax(dim=-1)
            targets = torch.tensor([f["label"] for f in facts], device=device)
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

# ==============================================================================
# 3. MAIN RUN ROUTINE
# ==============================================================================

def main():
    if FORCE_CPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Hardware Device: {device}")
    
    # Load SentenceTransformer locally (encoder will utilize target device)
    print("[*] Initializing SentenceTransformer encoder...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    
    # Static autocalibration for 384 input embedding dimensions
    print("[*] Calibrating static ScaleDistances for Layer 0...")
    shift_384, scale_384 = autocalibrate_scale_distances(384)
    
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    widths = [32, 64, 128, 256]
    capacity_steps = [5, 10, 20, 50, 100, 200, 400, 800]
    
    for u2_option in [True, False]:
        u2_label = "u2_Enabled" if u2_option else "u2_Ablated"
        print(f"\n====================================================")
        print(f"  RUNNING SWEEP: MESU Scaling with {u2_label}")
        print(f"====================================================")
        
        csv_path = f"kaggle_mesu_scaling_{u2_label.lower()}.csv"
        headers = [
            "seed", "experiment", "config", "alpha_decay", "u2_enabled", "num_facts",
            "recall", "forgetting", "drift", "variance_mean", "variance_min", "variance_max", "runtime_sec", "width"
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
        summary_data = {}
        
        for width in widths:
            print(f"\n[*] Calibrating and running for Width = {width}...")
            shift_w, scale_w = autocalibrate_scale_distances(width)
            summary_data[width] = {}
            
            for n_facts in capacity_steps:
                facts = generate_facts(n_facts)
                
                # Pre-embed using the sentence encoder in batches on GPU
                statements = [f["statement"] for f in facts]
                queries = [f["query"] for f in facts]
                
                with torch.no_grad():
                    # Batch encode on device (GPU/CPU)
                    emb_statements = encoder.encode(statements, convert_to_tensor=True, batch_size=256)
                    emb_queries = encoder.encode(queries, convert_to_tensor=True, batch_size=256)
                    
                    # Convert to list of unsqueezed tensors on the target device
                    embedded_statements = [emb_statements[i].unsqueeze(0).detach().clone().to(device) for i in range(n_facts)]
                    embedded_queries = [emb_queries[i].unsqueeze(0).detach().clone().to(device) for i in range(n_facts)]
                
                run_results = run_width_evaluation(
                    width, n_facts, seeds, facts, 
                    embedded_statements, embedded_queries, 
                    shift_w, scale_w, shift_384, scale_384, 
                    u2_enabled=u2_option, device=device
                )
                
                # Log raw results to CSV
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    for r in run_results:
                        writer.writerow({
                            "seed": r["seed"],
                            "experiment": f"kaggle_mesu_scaling_{u2_label.lower()}",
                            "config": f"width={width},num_facts={n_facts}",
                            "alpha_decay": 0.02,
                            "u2_enabled": u2_option,
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
                
                summary_data[width][n_facts] = (np.mean(rec_all), np.std(rec_all))
                print(f"  Width {width:3} | N {n_facts:3} | Recall: {np.mean(rec_all)*100:5.2f}% +/- {np.std(rec_all)*100:5.2f}% | Forgetting: {np.mean(forg_all)*100:5.2f}% +/- {np.std(forg_all)*100:5.2f}%")
        
        # Fit scaling laws
        fitting_results = {}
        print(f"\n====================================================")
        print(f"  Scaling Law Fitting: Recall(N) = A / N^alpha ({u2_label})")
        print(f"====================================================")
        print(f"{'Width':<6} | {'A (Scale)':<10} | {'alpha (Exp)':<12} | {'R^2 Quality':<10}")
        print("-" * 47)
        for width in widths:
            n_vals = capacity_steps
            rec_vals = [summary_data[width][n][0] for n in capacity_steps]
            A, alpha, r2 = fit_scaling_law(n_vals, rec_vals)
            fitting_results[width] = (A, alpha, r2)
            print(f"{width:<6} | {A:<10.4f} | {alpha:<12.4f} | {r2:<10.4f}")
            
        # Plot and save figures
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        for width in widths:
            rec_vals = [summary_data[width][n][0] for n in capacity_steps]
            plt.plot(capacity_steps, rec_vals, 'o-', label=f"W={width}")
        plt.xscale('log')
        plt.xlabel("N Facts")
        plt.ylabel("Recall")
        plt.title(f"Recall vs N ({u2_label})")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        for width in widths:
            rec_vals = np.array([summary_data[width][n][0] for n in capacity_steps])
            A, alpha, r2 = fitting_results[width]
            plt.scatter(capacity_steps, rec_vals + 1e-5)
            fitted = A * (np.array(capacity_steps) ** (-alpha))
            plt.plot(capacity_steps, fitted + 1e-5, '--', label=f"W={width} (alpha={alpha:.3f})")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Log(N)")
        plt.ylabel("Log(Recall)")
        plt.title(f"Log-Log Scale ({u2_label})")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = f"kaggle_mesu_scaling_{u2_label.lower()}_plots.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"[*] Visual plots saved to: {plot_path}")

if __name__ == "__main__":
    main()
