import torch
import torch.nn as nn
import numpy as np
import random
from mnc.layers import MNCLinear
from mnc.memory import MESUEngine
from pipeline import JournalPipeline

# =====================================================================
# 1. SETUP & AUTOCALIBRATION
# =====================================================================

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
    """Analytically derives scaling shifts and denominators from random sphere distance distribution."""
    set_seed(42)  # Use fixed seed for calibration stability
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

# Run autocalibration globally once to prevent seed resetting during model construction
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

# =====================================================================
# 2. ~5.2M PARAMETER DECODER-ONLY TRANSFORMER
# =====================================================================

class DecoderTransformer(nn.Module):
    """
    Standard GPT-style Decoder-Only Transformer (approx 5.17M parameters)
    which ingests sequences of MiniLM token-level embeddings [Seq_Len, 384].
    """
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1120, max_seq_len=64):
        super().__init__()
        self.input_proj = nn.Linear(384, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 10)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x shape: [1, Seq_Len, 384]
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        h = self.input_proj(x) + self.pos_encoder[:, :seq_len, :]
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.ln_f(h)
        # Classify at the last token representation
        logits = self.head(h[:, -1, :])
        return logits

# =====================================================================
# 3. EXPERIMENT STREAM DATA DEFINITIONS
# =====================================================================

FACTS = [
    {"statement": "The blue folder is in the third drawer.", "query": "Where is the blue folder kept?", "label": 0},
    {"statement": "The access code for the main server is 7734.", "query": "What's the main server's access code?", "label": 1},
    {"statement": "Sarah's strategy meeting was moved to Tuesday at 3 PM.", "query": "When did Sarah's strategy meeting get rescheduled to?", "label": 2},
    {"statement": "The backup generator requires unleaded fuel.", "query": "What kind of fuel does the backup generator need?", "label": 3},
    {"statement": "Sector 4 security cameras undergo maintenance at midnight.", "query": "When does Sector 4 camera maintenance happen?", "label": 4}
]

INTERFERENCE = [
    {"text": "The red folder is resting on the top desk.", "query": "Where is the red folder?", "label_std": 5, "label_hard": 0},
    {"text": "The access code for the guest wifi is 9912.", "query": "What is the guest wifi access code?", "label_std": 6, "label_hard": 1},
    {"text": "Someone left a coffee mug in the breakroom.", "query": "Where was the coffee mug left?", "label_std": 7, "label_hard": 2},
    {"text": "It is raining heavily outside today.", "query": "What is the weather like today?", "label_std": 8, "label_hard": 3}
]

# =====================================================================
# 4. TRAINING & EVALUATION FUNCTIONS
# =====================================================================

def run_experiment_mnc(seed, mode, pipeline, hardened=False, fact_steps=15, interference_steps=3, u2_enabled=True):
    set_seed(seed)
    model = make_mnc_model()
    
    if mode.startswith('mesu'):
        cond_mode = 'positive' if 'positive' in mode else 'negative'
        engine = MESUEngine(model, lr=1.0, sigma_prior=0.1, alpha_decay=0.02, conductance_mode=cond_mode, u2_enabled=u2_enabled)
        optimizer = None
    else:
        lr = float(mode.split('_')[1])
        engine = None
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Capture initial weights for trajectory tracking
    w_init = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    # Setup the stream
    stream = []
    for f in FACTS:
        stream.append({"text": f["statement"], "label": f["label"], "is_fact": True})
    for dist in INTERFERENCE:
        label = dist["label_hard"] if hardened else dist["label_std"]
        stream.append({"text": dist["text"], "label": label, "is_fact": False})

    # Save state after training on facts (Day 5)
    w_day5 = None
    day5_correct = 5

    model.train()
    for day_idx, item in enumerate(stream):
        # Save Day 5 weights right after training the last fact (Day 5)
        if day_idx == 5:
            w_day5 = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
            if engine is not None:
                u2_day5 = {n: engine.cascade_states[n + "_u2"].clone() for n, p in model.named_parameters() if (n + "_u2") in engine.cascade_states}
            
            # Measure Day 5 immediate recall baseline
            model.eval()
            day5_correct = 0
            with torch.no_grad():
                for f in FACTS:
                    qvec = pipeline.embed_sentence(f["query"])
                    logits = model(qvec)
                    pred = logits.argmax(dim=-1).item()
                    if pred == f["label"]:
                        day5_correct += 1
            model.train()

        vec = pipeline.embed_sentence(item["text"])
        num_steps = fact_steps if item["is_fact"] else interference_steps
        
        for step_idx in range(num_steps):
            noise = torch.randn_like(vec) * 0.05
            noisy_vec = vec + noise
            logits = model(noisy_vec)
            
            # Margin Contrastive Loss
            margin = 1.0
            loss = 0.0
            for idx_class in range(10):
                if idx_class != item["label"]:
                    loss += torch.clamp(logits[0, idx_class] - logits[0, item["label"]] + margin, min=0.0)
            
            loss.backward()
            
            if engine is not None:
                engine.step(loss.item())
                engine.zero_grad()
            else:
                optimizer.step()
                # Projection for SGD baseline
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if "W" in name and param.data.dim() == 2:
                            param.data.copy_(param.data / (param.data.norm(p=2, dim=1, keepdim=True) + 1e-8))
                optimizer.zero_grad()

    # Capture weights at Day 10
    w_day10 = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    if engine is not None:
        u2_day10 = {n: engine.cascade_states[n + "_u2"].clone() for n, p in model.named_parameters() if (n + "_u2") in engine.cascade_states}

    # Evaluate Day 10 Recall
    model.eval()
    fact_correct = 0
    with torch.no_grad():
        for f in FACTS:
            qvec = pipeline.embed_sentence(f["query"])
            logits = model(qvec)
            pred = logits.argmax(dim=-1).item()
            if pred == f["label"]:
                fact_correct += 1

    dist_correct = 0
    displacement_count = 0
    if hardened:
        with torch.no_grad():
            for dist in INTERFERENCE:
                # Query the distractor
                qvec = pipeline.embed_sentence(dist["query"])
                logits_dist = model(qvec)
                pred_dist = logits_dist.argmax(dim=-1).item()
                if pred_dist == dist["label_hard"]:
                    dist_correct += 1
                
                # Check displacement: if distractor maps to label, but fact does not
                fact_qvec = pipeline.embed_sentence(FACTS[dist["label_hard"]]["query"])
                pred_fact = model(fact_qvec).argmax(dim=-1).item()
                if pred_dist == dist["label_hard"] and pred_fact != dist["label_hard"]:
                    displacement_count += 1

    # Trajectory Telemetry
    telemetry = {}
    if w_day5 is not None:
        # Distance calculation
        def get_dist(dict1, dict2):
            total_d = 0.0
            for k in dict1.keys():
                total_d += torch.norm(dict1[k] - dict2[k], p=2).item()
            return total_d

        telemetry["d_W10_W5"] = get_dist(w_day10, w_day5)
        telemetry["d_W10_Winit"] = get_dist(w_day10, w_init)
        if engine is not None:
            telemetry["d_u210_W5"] = get_dist(u2_day10, w_day5)

    return fact_correct, dist_correct, displacement_count, telemetry, day5_correct

# ---------------------------------------------------------------------

def run_experiment_transformer(seed, pipeline, lr=1e-4, fact_steps=15, interference_steps=3):
    set_seed(seed)
    model = DecoderTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    stream = []
    for f in FACTS:
        stream.append({"text": f["statement"], "label": f["label"], "is_fact": True})
    for dist in INTERFERENCE:
        stream.append({"text": dist["text"], "label": dist["label_std"], "is_fact": False})

    model.train()
    for day_idx, item in enumerate(stream):
        # Retrieve token-level MiniLM embeddings [1, Seq_Len, 384]
        token_vecs = pipeline.embed_tokens(item["text"])
        num_steps = fact_steps if item["is_fact"] else interference_steps
        
        for step_idx in range(num_steps):
            # Target output is cross entropy logit
            logits = model(token_vecs)
            target = torch.tensor([item["label"]], dtype=torch.long)
            loss = loss_fn(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    fact_correct = 0
    with torch.no_grad():
        for f in FACTS:
            token_vecs = pipeline.embed_tokens(f["query"])
            logits = model(token_vecs)
            pred = logits.argmax(dim=-1).item()
            if pred == f["label"]:
                fact_correct += 1

    return fact_correct

# =====================================================================
# 5. EXECUTION & REPORT GENERATION
# =====================================================================

if __name__ == "__main__":
    print("[*] Initializing local SentenceTransformer pipeline...")
    pipeline = JournalPipeline()
    seeds = list(range(10))

    # -----------------------------------------------------------------
    # STUDY 1: Baseline Sweep (Standard Protocol with Autocalibrated Scale)
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("STUDY 1: BASELINE SWEEP (STANDARD PROTOCOL WITH AUTOCALIBRATION)")
    print("="*70)
    modes = ['mesu_negative', 'mesu_positive', 'sgd_1.0', 'sgd_0.1']
    study1_results = {m: {"fc": [], "day5": []} for m in modes}
    for seed in seeds:
        for m in modes:
            fc, _, _, _, day5_c = run_experiment_mnc(seed, m, pipeline, hardened=False)
            study1_results[m]["fc"].append(fc)
            study1_results[m]["day5"].append(day5_c)

    for m in modes:
        fcs = study1_results[m]["fc"]
        day5s = study1_results[m]["day5"]
        mean_fc = np.mean(fcs)
        mean_day5 = np.mean(day5s)
        ratio_of_means = (100.0 * mean_fc / mean_day5) if mean_day5 > 0 else 0.0
        print(f"  {m:<15} | Mean Recall: {mean_fc:.2f}/5 (Day 5: {mean_day5:.2f}/5) | Recall Ratio (Day10/Day5): {ratio_of_means:.2f}% | Range: [{np.min(fcs)}, {np.max(fcs)}]")

    # -----------------------------------------------------------------
    # STUDY 2: Hardened Interference Test (Shared Output Labels)
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("STUDY 2: HARDENED INTERFERENCE TEST (SHARED OUTPUT LABELS)")
    print("="*70)
    study2_results = {m: {"facts": [], "dists": [], "displaced": []} for m in ['mesu_negative', 'sgd_0.1']}
    for seed in seeds:
        for m in ['mesu_negative', 'sgd_0.1']:
            fc, dc, disp, _, _ = run_experiment_mnc(seed, m, pipeline, hardened=True)
            study2_results[m]["facts"].append(fc)
            study2_results[m]["dists"].append(dc)
            study2_results[m]["displaced"].append(disp)

    for m in ['mesu_negative', 'sgd_0.1']:
        res = study2_results[m]
        print(f"  {m:<15} | Fact Recall: {np.mean(res['facts']):.2f}/5 | Distractor Recall: {np.mean(res['dists']):.2f}/4 | Displaced Facts: {np.mean(res['displaced']):.2f}")

    # -----------------------------------------------------------------
    # STUDY 3: Step Symmetry Ablation
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("STUDY 3: STEP BUDGET SYMMETRY ABLATION")
    print("="*70)
    step_configs = [
        {"name": "Asymmetric (15/3)", "fact": 15, "interf": 3},
        {"name": "Symmetric Short (5/5)", "fact": 5, "interf": 5},
        {"name": "Symmetric Long (15/15)", "fact": 15, "interf": 15}
    ]
    study3_results = {m: {cfg["name"]: [] for cfg in step_configs} for m in ['mesu_negative', 'sgd_0.1']}
    
    for seed in seeds:
        for m in ['mesu_negative', 'sgd_0.1']:
            for cfg in step_configs:
                fc, _, _, _, _ = run_experiment_mnc(seed, m, pipeline, hardened=False, fact_steps=cfg["fact"], interference_steps=cfg["interf"])
                study3_results[m][cfg["name"]].append(fc)

    for m in ['mesu_negative', 'sgd_0.1']:
        print(f"  Optimizer: {m}")
        for cfg in step_configs:
            scores = study3_results[m][cfg["name"]]
            print(f"    {cfg['name']:<22} | Mean Recall: {np.mean(scores):.2f}/5 | Std: {np.std(scores):.2f}")

    # -----------------------------------------------------------------
    # STUDY 4: u2 Telemetry and Restorative Pull Ablation
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("STUDY 4: U2 CASCADE TELEMETRY & RESTORATIVE PULL ABLATION")
    print("="*70)
    study4_results = {
        "enabled": {"scores": [], "d_W10_W5": [], "d_u210_W5": [], "d_W10_Winit": []},
        "disabled": {"scores": [], "d_W10_W5": [], "d_u210_W5": [], "d_W10_Winit": []}
    }
    
    for seed in seeds:
        # Enabled
        fc, _, _, tel, _ = run_experiment_mnc(seed, 'mesu_negative', pipeline, u2_enabled=True)
        study4_results["enabled"]["scores"].append(fc)
        study4_results["enabled"]["d_W10_W5"].append(tel["d_W10_W5"])
        study4_results["enabled"]["d_u210_W5"].append(tel["d_u210_W5"])
        study4_results["enabled"]["d_W10_Winit"].append(tel["d_W10_Winit"])

        # Disabled
        fc, _, _, tel, _ = run_experiment_mnc(seed, 'mesu_negative', pipeline, u2_enabled=False)
        study4_results["disabled"]["scores"].append(fc)
        study4_results["disabled"]["d_W10_W5"].append(tel["d_W10_W5"])
        study4_results["disabled"]["d_u210_W5"].append(tel["d_u210_W5"])
        study4_results["disabled"]["d_W10_Winit"].append(tel["d_W10_Winit"])

    for k in ["enabled", "disabled"]:
        res = study4_results[k]
        print(f"  u2 restorative pull {k.upper()}:")
        print(f"    Mean Recall Accuracy      : {np.mean(res['scores']):.2f}/5")
        print(f"    Parameter Drift (W10-W5)  : {np.mean(res['d_W10_W5']):.4f}")
        print(f"    Cascade Drift (u2_10-W5)  : {np.mean(res['d_u210_W5']):.4f}")
        print(f"    Total Weight Shift (W10-W0): {np.mean(res['d_W10_Winit']):.4f}")

    # -----------------------------------------------------------------
    # STUDY 5: 5.2M Parameter Decoder-Only Transformer Baseline
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("STUDY 5: TRANSFORMER CONTROL COMPARISON")
    print("="*70)
    
    print("[*] Tuning Transformer control hyperparameters on Seed 0...", flush=True)
    best_lr = 1e-4
    best_steps = 15
    best_score = -1
    
    for lr in [1e-4, 5e-4, 1e-3, 3e-3]:
        for steps in [15, 30, 50]:
            score = run_experiment_transformer(0, pipeline, lr=lr, fact_steps=steps)
            if score > best_score:
                best_score = score
                best_lr = lr
                best_steps = steps
                
    print(f"  Selected optimal parameters: LR = {best_lr:.5f}, Fact Steps = {best_steps} (Seed 0 Tuning Score: {best_score}/5)", flush=True)
    
    trans_scores = []
    for seed in seeds:
        fc = run_experiment_transformer(seed, pipeline, lr=best_lr, fact_steps=best_steps)
        trans_scores.append(fc)
        print(f"  Seed {seed} Transformer: {fc}/5")
    
    print(f"\n  Transformer Control Mean: {np.mean(trans_scores):.2f}/5 | Std: {np.std(trans_scores):.2f} | Range: [{np.min(trans_scores)}, {np.max(trans_scores)}]")
    print(f"  MNC (mesu_negative) Mean : {np.mean(study1_results['mesu_negative']['fc']):.2f}/5 | Std: {np.std(study1_results['mesu_negative']['fc']):.2f}")
