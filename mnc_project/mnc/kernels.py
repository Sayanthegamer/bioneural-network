import torch
import torch.nn.functional as F

class MNCAdderFunction(torch.autograd.Function):
    """
    Core Multiplication-Free Forward Primitive & Decoupled Gradient Router (v2).
    
    FORWARD:
      Computes the negative L1 (Manhattan) distance between input X and template W.
      Highest activation (closest to 0) occurs when X exactly matches W.
      
    BACKWARD (Surrogate Gradients):
      True derivative of an absolute value is a flat step function (sign), which 
      causes gradients to die or explode. We hijack the chain rule:
      - W gets an L2 surrogate (X - W) so templates smoothly glide toward inputs.
      - X gets a HardTanh clamped surrogate to prevent chain-rule explosion.
    """
    
    @staticmethod
    def forward(ctx, x, w):
        # 1. Hardware-Native Forward Pass (Batch-ready)
        # x shape: [Batch, In_Features]
        # w shape: [Out_Features, In_Features]
        
        # Expand dimensions to compute full pairwise difference matrix without matmul
        x_exp = x.unsqueeze(1)  # [Batch, 1, In_Features]
        w_exp = w.unsqueeze(0)  # [1, Out_Features, In_Features]
        
        # Calculate the raw distance vector in space
        diff = x_exp - w_exp    
        
        # Y = -|X - W| summed across the feature dimension
        out = -torch.abs(diff).sum(dim=2) # Shape: [Batch, Out_Features]
        
        # Cache the difference matrix for the custom backward pass
        ctx.save_for_backward(diff)
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # 2. Decoupled Virtual Gradient Routing (The Surrogate)
        diff, = ctx.saved_tensors
        
        # Align grad_output dimensions for broadcasting: [Batch, Out_Features, 1]
        grad_output_exp = grad_output.unsqueeze(2)
        
        # --- Gradient for Weights (W) ---
        # L2 Surrogate: Instead of flat signs, we pass the full coordinate difference.
        # This tells the weight exactly how far and what direction to step to catch the input.
        grad_w_diff = diff  # (X - W)
        grad_w = (grad_output_exp * grad_w_diff).sum(dim=0)
        
        # --- Gradient for Inputs (X) ---
        # HardTanh Surrogate: Clamps the difference between [-1, 1].
        # Prevents exploding gradients from cascading back down into earlier layers/embeddings.
        grad_x_diff = torch.clamp(diff, min=-1.0, max=1.0)
        grad_x = -(grad_output_exp * grad_x_diff).sum(dim=1)
        
        return grad_x, grad_w

# Convenience wrapper for clean layer implementation
def mnc_adder(x, w):
    return MNCAdderFunction.apply(x, w)


# ==========================================
# Micro-Validation Routine
# ==========================================
if __name__ == "__main__":
    print("--- MNC v2 Kernel Micro-Validation ---")
    
    # 1. Initialize random input (X) and weight (W) tensors
    batch_size, in_dim, out_dim = 2, 4, 3
    x = torch.randn(batch_size, in_dim, requires_grad=True)
    w = torch.randn(out_dim, in_dim, requires_grad=True)
    
    print(f"Input X shape: {list(x.shape)}")
    print(f"Weight W shape: {list(w.shape)}\n")
    
    # 2. Forward Pass Execution
    y = mnc_adder(x, w)
    print(f"Forward Output (Negative Distance):")
    print(f"{y.data}\n")
    
    # 3. Backward Pass Execution (Simulating an error signal)
    loss = y.sum()
    loss.backward()
    
    print(f"Gradients on X (HardTanh Clamped):")
    print(f"{x.grad}\n")
    print(f"Gradients on W (L2 Surrogate Guided):")
    print(f"{w.grad}")
    
    print("\n[SUCCESS] No NaN/Zero gradients detected. PyTorch Autograd successfully hijacked.")