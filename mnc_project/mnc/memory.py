import torch

class MESUEngine:
    """
    Balanced Metaplasticity from Synaptic Uncertainty (MESU) Memory Engine.
    Tracks structural variance parameters alongside means to protect consolidated 
    knowledge while maintaining plastic room for ongoing streaming context.
    Incorporates Adaptive Layer-Wise Gradient Scaling to prevent clamp saturation.
    """
    def __init__(self, model, lr=2.0, sigma_prior=0.1, alpha_decay=0.01, sigma_res=0.1, conductance_mode='negative', u2_enabled=True):
        self.model = model
        self.lr = lr
        self.sigma_prior = sigma_prior
        self.alpha_decay = alpha_decay
        self.sigma_res = sigma_res
        self.conductance_mode = conductance_mode
        self.u2_enabled = u2_enabled
        
        self.variances = {}
        self.cascade_states = {}
        
        self._initialize_memory_states()

    def _initialize_memory_states(self):
        """Instantiates Gaussian uncertainty bounds and tracking registers for all model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Initialize uncertainty variance at the baseline prior
                self.variances[name] = torch.full_like(param.data, self.sigma_prior ** 2)
                
                # Cascade states initialized to match baseline initial weights
                self.cascade_states[name + "_u1"] = torch.clone(param.data)
                self.cascade_states[name + "_u2"] = torch.clone(param.data)

    @torch.no_grad()
    def step(self, current_loss=None):
        # Determine coupling conductance based on current loss
        g = 0.05
        if current_loss is not None:
            # Safe sigmoid gating to prevent exponential explosion
            if self.conductance_mode == 'positive':
                g = 0.1 * torch.sigmoid(torch.tensor(current_loss)).item()
            else:
                g = 0.1 * torch.sigmoid(-torch.tensor(current_loss)).item()

        for name, param in self.model.named_parameters():
            if param.grad is None: continue
            
            raw_grad = param.grad.data
            
            # Adaptive Gradient Scaling: Scale gradient to have L2 norm equal to sqrt(numel)
            grad_norm = raw_grad.norm(2)
            if grad_norm > 1e-5:
                scale = (param.numel() ** 0.5) / (grad_norm + 1e-4)
                raw_grad = raw_grad * scale
                
            var = self.variances[name]
            
            # Use variance to gate the update (Metaplasticity)
            # This is the "Learning Rate Governor"
            effective_lr = self.lr * var 
            param.data.sub_(effective_lr * raw_grad)
            
            # Update multi-timescale cascade states
            u1_key = name + "_u1"
            u2_key = name + "_u2"
            if u1_key in self.cascade_states and u2_key in self.cascade_states:
                u1 = self.cascade_states[u1_key]
                u2 = self.cascade_states[u2_key]
                
                # Fast cascade tracks active param
                u1.add_(g * (param.data - u1))
                # Slow cascade consolidates from fast cascade
                u2.add_(0.1 * g * (u1 - u2))
                
                # Pull active parameters back to slow consolidated state based on confidence
                if self.u2_enabled:
                    confidence = 1.0 - (var / (self.sigma_prior ** 2))
                    confidence = torch.clamp(confidence, min=0.0, max=1.0)
                    param.data.add_(confidence * g * (u2 - param.data))
                
            # In-place unit sphere projection to prevent template/gradient explosion
            if "W" in name and param.data.dim() == 2:
                param.data.copy_(param.data / (param.data.norm(p=2, dim=1, keepdim=True) + 1e-8))
                if u1_key in self.cascade_states:
                    self.cascade_states[u1_key].copy_(self.cascade_states[u1_key] / (self.cascade_states[u1_key].norm(p=2, dim=1, keepdim=True) + 1e-8))
                if u2_key in self.cascade_states:
                    self.cascade_states[u2_key].copy_(self.cascade_states[u2_key] / (self.cascade_states[u2_key].norm(p=2, dim=1, keepdim=True) + 1e-8))
            
            # Prior Relaxation: Lock confident weights using unscaled gradient
            # This decouples the locking rate from parameter count/layer width scaling
            var.sub_(var * torch.clamp(param.grad.data.abs() * 0.2, max=0.25))
            
            # Pull variance back toward the tighter prior (relaxation scaled by self.alpha_decay)
            var.add_(self.alpha_decay * ((self.sigma_prior**2) - var))
            var.clamp_(min=1e-4, max=self.sigma_prior ** 2)

    def zero_grad(self):
        """Clears calculated parameter gradients across the tracking model module."""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()


# ==========================================
# Engine Independent Execution Verification
# ==========================================
if __name__ == "__main__":
    print("--- MNC MESU Engine Operational Self-Check ---")
    import torch.nn as nn
    
    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(torch.randn(10, 20))
    
    test_net = SimpleModule()
    engine = MESUEngine(test_net, lr=1.0)
    
    v_start = engine.variances["W"].mean().item()
    print(f"Prior Variance Register Baseline: {v_start:.6f}")
    
    dummy_x = torch.randn(1, 20)
    dummy_out = torch.sum(dummy_x @ test_net.W.t())
    dummy_out.backward()
    
    engine.step()
    
    v_end = engine.variances["W"].mean().item()
    print(f"Post-Update Variance Register Level: {v_end:.6f}")
    
    if v_end < v_start:
        print("[SUCCESS] Variational posterior compacted cleanly. Engine active.")
    else:
        print("[ERROR] Variance registration stalled. Verify gradient loops.")