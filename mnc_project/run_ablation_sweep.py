import torch
import numpy as np
import random
from mnc.layers import MNCLinear
from mnc.memory import MESUEngine
from pipeline import JournalPipeline

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiment(seed, mode, pipeline):
    set_seed(seed)
    
    class ScaleDistances(torch.nn.Module):
        def __init__(self, shift, scale):
            super().__init__()
            self.shift = shift
            self.scale = scale
        def forward(self, x):
            return (x + self.shift) / self.scale
            
    mnc_model = torch.nn.Sequential(
        MNCLinear(384, 32),
        ScaleDistances(22.0, 2.0),
        torch.nn.Tanh(),
        MNCLinear(32, 10),
        ScaleDistances(6.4, 1.0)
    )
    
    if mode.startswith('mesu'):
        cond_mode = 'positive' if 'positive' in mode else 'negative'
        engine = MESUEngine(mnc_model, lr=1.0, sigma_prior=0.1, alpha_decay=0.02, conductance_mode=cond_mode)
        optimizer = None
    elif mode.startswith('sgd'):
        lr = float(mode.split('_')[1])
        engine = None
        optimizer = torch.optim.SGD(mnc_model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    FACTS = [
        {"statement": "The blue folder is in the third drawer.", "query": "Where is the blue folder kept?", "label": 0},
        {"statement": "The access code for the main server is 7734.", "query": "What's the main server's access code?", "label": 1},
        {"statement": "Sarah's strategy meeting was moved to Tuesday at 3 PM.", "query": "When did Sarah's strategy meeting get rescheduled to?", "label": 2},
        {"statement": "The backup generator requires unleaded fuel.", "query": "What kind of fuel does the backup generator need?", "label": 3},
        {"statement": "Sector 4 security cameras undergo maintenance at midnight.", "query": "When does Sector 4 camera maintenance happen?", "label": 4}
    ]

    INTERFERENCE = [
        "The red folder is resting on the top desk.",
        "Someone left a coffee mug in the breakroom.",
        "The access code for the guest wifi is 9912.",
        "It is raining heavily outside today."
    ]

    JOURNAL_LINES = []
    for f in FACTS:
        JOURNAL_LINES.append({"text": f["statement"], "label": f["label"]})
    for idx, text in enumerate(INTERFERENCE):
        JOURNAL_LINES.append({"text": text, "label": 5 + idx})

    mnc_model.train()
    
    for i, item in enumerate(JOURNAL_LINES):
        text = item["text"]
        label = item["label"]
        vec = pipeline.embed_sentence(text)
        
        num_steps = 15 if label < 5 else 3
        for step_idx in range(num_steps):
            noise = torch.randn_like(vec) * 0.05
            noisy_vec = vec + noise
            
            logits = mnc_model(noisy_vec)
            
            # DECOUPLED BOUNDARY LOSS
            margin_wrong = 1.0
            margin_true = 0.2
            
            # 1. Target Pull: Pull the correct class to an absolute safe zone
            loss = torch.clamp(-logits[0, label] - margin_true, min=0.0)
            
            # 2. Intrusion Penalty: Push wrong classes away ONLY if they cross the absolute margin
            for idx_class in range(10):
                if idx_class != label:
                    loss += torch.clamp(logits[0, idx_class] + margin_wrong, min=0.0)
            
            loss.backward()
            
            if engine is not None:
                engine.step(loss.item())
                engine.zero_grad()
            else:
                optimizer.step()
                # Apply weight projection to the unit sphere to keep templates bounded
                with torch.no_grad():
                    for name, param in mnc_model.named_parameters():
                        if "W" in name and param.data.dim() == 2:
                            param.data.copy_(param.data / (param.data.norm(p=2, dim=1, keepdim=True) + 1e-8))
                optimizer.zero_grad()

    mnc_model.eval()
    correct = 0
    with torch.no_grad():
        for f in FACTS:
            qvec = pipeline.embed_sentence(f["query"])
            logits = mnc_model(qvec)
            pred = logits.argmax(dim=-1).item()
            if pred == f["label"]:
                correct += 1
                
    return correct

if __name__ == "__main__":
    print("--- Running Ablation Sweep & Seed Sweep (10 Seeds) ---")
    pipeline = JournalPipeline()
    
    modes = ['mesu_negative', 'mesu_positive', 'sgd_1.0', 'sgd_0.1', 'sgd_0.01']
    seeds = list(range(10))
    
    results = {m: [] for m in modes}
    
    for seed in seeds:
        print(f"\nSeed {seed}:")
        for mode in modes:
            score = run_experiment(seed, mode, pipeline)
            results[mode].append(score)
            print(f"  {mode:<15}: {score}/5")
            
    print("\n" + "="*40)
    print("SUMMARY STATISTICS (Correct recall out of 5)")
    print("="*40)
    for mode in modes:
        scores = results[mode]
        mean = np.mean(scores)
        std = np.std(scores)
        min_s = np.min(scores)
        max_s = np.max(scores)
        print(f"{mode:<15} | Mean: {mean:.2f}/5 | Std: {std:.2f} | Range: [{min_s}, {max_s}]")
