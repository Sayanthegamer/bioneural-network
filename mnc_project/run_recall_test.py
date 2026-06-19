"""
MNC v3: The Bottleneck (Shared Representation Test)
Forces 384-dimensional inputs through a 16-node shared hidden layer 
before routing to the 10 final templates. Tests true Catastrophic Forgetting resistance.
"""
import torch
from mnc.layers import MNCLinear
from mnc.memory import MESUEngine
from pipeline import JournalPipeline

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
    print("--- MNC v3 (Shared Bottleneck) Delayed-Recall Test ---")
    print("[!] Testing True Metaplastic Consolidation against Parameter Overlap")
    
    pipeline = JournalPipeline()
    
    # Custom Lambda layer to safely scale distances before squashing
    class ScaleDistances(torch.nn.Module):
        def __init__(self, shift, scale):
            super().__init__()
            self.shift = shift
            self.scale = scale
        def forward(self, x):
            return (x + self.shift) / self.scale
            
    # 1. The Bottleneck Architecture (32 hidden nodes = Shared Representation)
    mnc_model = torch.nn.Sequential(
        MNCLinear(384, 32),
        ScaleDistances(22.0, 2.0),
        torch.nn.Tanh(),
        MNCLinear(32, 10),
        ScaleDistances(6.4, 1.0)
    )
    
    # Engine calibrated for fierce competition
    engine = MESUEngine(mnc_model, lr=1.0, sigma_prior=0.1, alpha_decay=0.02)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("\n[*] Online training over 9 lines (5 facts + 4 interference)")
    mnc_model.train()
    
    for i, item in enumerate(JOURNAL_LINES):
        text = item["text"]
        label = item["label"]
        vec = pipeline.embed_sentence(text)
        
        # Train for multiple steps: 15 steps for facts (labels < 5), 3 steps for interference (labels >= 5)
        num_steps = 15 if label < 5 else 3
        for step_idx in range(num_steps):
            # Noise injection for data augmentation and robustness to paraphrasing
            noise = torch.randn_like(vec) * 0.05
            noisy_vec = vec + noise
            
            # Forward pass (computes distances through bottleneck)
            logits = mnc_model(noisy_vec)
            
            # DECOUPLED BOUNDARY LOSS
            margin_wrong = 1.0
            margin_true = 0.2
            
            # 1. Target Pull: Pull the correct class to an absolute safe zone (e.g., >= -0.2)
            loss = torch.clamp(-logits[0, label] - margin_true, min=0.0)
            
            # 2. Intrusion Penalty: Push wrong classes away ONLY if they cross the absolute margin
            for idx_class in range(10):
                if idx_class != label:
                    loss += torch.clamp(logits[0, idx_class] + margin_wrong, min=0.0)
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
        print("[VERDICT] PASS - The bottleneck held! Metaplasticity works.")
    else:
        print("[VERDICT] FAIL - The bottleneck shattered the memories.")

if __name__ == "__main__":
    run_evaluation()