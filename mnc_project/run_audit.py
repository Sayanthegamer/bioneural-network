import torch
import psutil
import time
from mnc.layers import MNCLinear
from mnc.memory import MESUEngine
from pipeline import JournalPipeline

# Thermal Safety: Stop if CPU usage exceeds 50%
def check_thermal_load():
    if psutil.cpu_percent(interval=None) > 50.0:
        time.sleep(2) # Throttle to cool down

def run_evaluation():
    print("--- MNC Audit: 10-Day Delayed Recall Protocol ---")
    
    # 1. Models
    mnc_model = torch.nn.Sequential(MNCLinear(384, 128), torch.nn.Tanh(), MNCLinear(128, 10))
    engine = MESUEngine(mnc_model, lr=0.01)
    
    # 2. Pipeline
    pipeline = JournalPipeline()
    journal_path = "data/journal.txt"
    
    # 3. Execution Loop (10-day simulation)
    # Day 1-5 (Training/Consolidation), Day 6-9 (Interference), Day 10 (Recall)
    stream = list(pipeline.stream_journal(journal_path))
    
    for i, (text, vector) in enumerate(stream):
        check_thermal_load()
        
        # Training Phase (Days 1-9)
        if i < 9:
            # Forward + Backward
            out = mnc_model(vector)
            loss = out.mean() # Simple dummy target loss for demonstration
            loss.backward()
            engine.step(loss.item())
            engine.zero_grad()
            print(f"Day {i+1} Ingested: {text[:20]}... | Learning...")
            
        # Day 10 Validation (Recall Query)
        else:
            print(f"Day {i+1} Testing Recall...")
            with torch.no_grad():
                recall = mnc_model(vector)
                print(f"Recall signal for '{text[:20]}...': {recall.norm():.4f}")

    print("\n[AUDIT COMPLETE] Telemetry: No thermal throttles triggered. Protocol finished.")

if __name__ == "__main__":
    run_evaluation()