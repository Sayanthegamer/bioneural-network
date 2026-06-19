"""
MNC v4: The Bottleneck (Shared Representation Test)
Forces 384-dimensional inputs through a 32-node shared hidden layer 
before routing to the 10 final templates. Tests true Catastrophic Forgetting resistance.

Uses CrossEntropyLoss instead of margin losses. CrossEntropyLoss has a critical
property for bottleneck architectures: its logit gradients always sum to exactly
zero (sum of p_j - y_j = 0). This means the shared Layer 0 receives perfectly
balanced forces — no net destructive push that collapses the bottleneck.
"""
import torch
import torch.nn as nn
import numpy as np
import random
from mnc.layers import MNCLinear
from mnc.memory import MESUEngine
from pipeline import JournalPipeline

# =====================================================================
# AUTOCALIBRATION (matches run_comprehensive_validation.py)
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
    set_seed(42)  # Fixed seed for calibration stability
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

# Run autocalibration globally once
SHIFT0, SCALE0 = autocalibrate_scale_distances(384)
SHIFT1, SCALE1 = autocalibrate_scale_distances(32)

FACTS = [
    {"statement": "The blue folder is in the third drawer.", "query": "Where is the blue folder kept?", "answer": "third drawer", "label": 0},
    {"statement": "The access code for the main server is 7734.", "query": "What's the main server's access code?", "answer": "7734", "label": 1},
    {"statement": "Sarah's strategy meeting was moved to Tuesday at 3 PM.", "query": "When did Sarah's strategy meeting get rescheduled to?", "answer": "Tuesday at 3 PM", "label": 2},
    {"statement": "The backup generator requires unleaded fuel.", "query": "What kind of fuel does the backup generator need?", "answer": "unleaded fuel", "label": 3},
    {"statement": "Sector 4 security cameras undergo maintenance at midnight.", "query": "When does Sector 4 camera maintenance happen?", "answer": "midnight", "label": 4}
]

INTERFERENCE = [
    "The red folder is resting on the top desk.",
    "Someone left a coffee mug in the breakroom.",
    "The access code for the guest wifi is 9912.",
    "It is raining heavily outside today."
]

# Build the sequential stream
JOURNAL_LINES = []
for f in FACTS:
    JOURNAL_LINES.append({"text": f["statement"], "label": f["label"]})
for idx, text in enumerate(INTERFERENCE):
    JOURNAL_LINES.append({"text": text, "label": 5 + idx})

def run_evaluation():
    print("--- MNC v4 (Shared Bottleneck + CrossEntropy) Delayed-Recall Test ---")
    print("[!] Testing True Metaplastic Consolidation against Parameter Overlap")
    print(f"[*] Autocalibrated: Layer0 shift={SHIFT0:.2f} scale={SCALE0:.2f} | Layer1 shift={SHIFT1:.2f} scale={SCALE1:.2f}")
    
    pipeline = JournalPipeline()
    
    # The Bottleneck Architecture (32 hidden nodes = Shared Representation)
    # Uses autocalibrated ScaleDistances instead of hardcoded constants
    mnc_model = nn.Sequential(
        MNCLinear(384, 32),
        ScaleDistances(SHIFT0, SCALE0),
        nn.Tanh(),
        MNCLinear(32, 10),
        ScaleDistances(SHIFT1, SCALE1)
    )
    
    # Engine calibrated for fierce competition
    engine = MESUEngine(mnc_model, lr=1.0, sigma_prior=0.1, alpha_decay=0.02)
    
    # CrossEntropyLoss: gradients on logits sum to zero (∑(p_j - y_j) = 0)
    # This means the shared bottleneck layer gets balanced forces — no net destruction
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n[*] Online training over {len(JOURNAL_LINES)} lines (5 facts + 4 interference)")
    mnc_model.train()
    
    for i, item in enumerate(JOURNAL_LINES):
        text = item["text"]
        label = item["label"]
        vec = pipeline.embed_sentence(text)
        target = torch.tensor([label])
        
        # Force symmetric training (15 steps for both facts and interference)
        # to expose the true limitation of the L1 lockbox and the variance ratchet.
        # Running symmetric training causes recall to collapse, proving that MESU 
        # cannot handle equal-weight sequential interference without a replay buffer.
        num_steps = 15
        for step_idx in range(num_steps):
            # Noise injection for data augmentation and robustness to paraphrasing
            noise = torch.randn_like(vec) * 0.05
            noisy_vec = vec + noise
            
            # Forward pass (computes distances through bottleneck)
            logits = mnc_model(noisy_vec)
            
            # CrossEntropyLoss — zero-sum gradients protect the shared bottleneck
            loss = loss_fn(logits, target)
            print(f"  Step {step_idx+1}: loss={loss.item():.4f}")
            
            # Backprop & MESU Update
            loss.backward()
            if step_idx == 0 or step_idx == num_steps - 1:
                g0 = mnc_model[0].W.grad.abs().mean().item() if mnc_model[0].W.grad is not None else 0.0
                g3 = mnc_model[3].W.grad.abs().mean().item() if mnc_model[3].W.grad is not None else 0.0
                print(f"    Step {step_idx+1} gradients: 0.W={g0:.6f}, 3.W={g3:.6f}")
            engine.step(loss.item())
            engine.zero_grad()
        
        # Check prediction after training on this sample
        logits = mnc_model(vec)
        pred = logits.argmax(dim=-1).item()
        var_0 = engine.variances['0.W'].mean().item()
        var_2 = engine.variances['3.W'].mean().item()
        print(f"Day {i+1}: '{text[:40]:<40}' | label={label} pred={pred} loss={loss.item():.4f} | var_0={var_0:.6f} var_2={var_2:.6f}")

    print("\n[*] Day 10 — querying with PARAPHRASED questions (not seen verbatim in training)")
    mnc_model.eval()
    correct = 0
    with torch.no_grad():
        for f in FACTS:
            qvec = pipeline.embed_sentence(f["query"])
            logits = mnc_model(qvec)
            pred = logits.argmax(dim=-1).item()
            
            is_correct = (pred == f["label"])
            correct += int(is_correct)
            status = "CORRECT" if is_correct else "WRONG"
            
            print(f"  Q: {f['query']}")
            print(f"     Expected {f['label']}, got {pred} -> {status}")

    pct = 100.0 * correct / len(FACTS)
    print(f"\n[RESULT] {correct}/{len(FACTS)} correct ({pct:.0f}%)")
    
    if pct >= 80:
        print("[VERDICT] PASS - Unexpectedly passed under symmetric training. (Check random seed state).")
    else:
        print("[VERDICT] FAIL - The bottleneck shattered. As predicted by the mathematical limits of the L1 coordinate dispute,")
        print("                 symmetric interference forces either catastrophic remembering or catastrophic forgetting.")

if __name__ == "__main__":
    run_evaluation()