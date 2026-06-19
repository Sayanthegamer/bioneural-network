import torch
import torch.nn as nn
from mnc.kernels import mnc_adder

class MNCLinear(nn.Module):
    """
    Multiplication-Free Metaplastic Neuro-Channel Linear Layer (v2).
    
    Operates as an Associative Distance Bank. Computes the negative L1 distance 
    between input feature vectors and a set of learned spatial templates (W).
    Bypasses standard matrix multiplication entirely.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Spatial Template Matrix (W)
        # Represents coordinates in the feature space.
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Biases
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize templates on the unit sphere to match sentence embedding scale
        nn.init.normal_(self.W, mean=0.0, std=1.0)
        with torch.no_grad():
            self.W.data = self.W.data / (self.W.data.norm(p=2, dim=1, keepdim=True) + 1e-8)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Input x shape: [Batch, In_Features]
        Output shape: [Batch, Out_Features]
        """
        # Execute custom hardware-native distance calculation
        out = mnc_adder(x, self.W)
        
        # Apply bias
        out = out + self.bias
        
        return out


# ==========================================
# Layer Micro-Validation Routine
# ==========================================
if __name__ == "__main__":
    print("--- MNC Linear Layer (v2) Micro-Validation ---")
    
    in_dim = 384
    out_dim = 10
    
    # 1. Instantiate the layer
    layer = MNCLinear(in_dim, out_dim)
    print(f"Initialized Layer: {layer}")
    
    # 2. Simulate a standard sentence embedding vector
    x_sample = torch.randn(1, in_dim, requires_grad=True)
    
    # 3. Forward Pass
    y_out = layer(x_sample)
    print(f"Output shape: {list(y_out.shape)} (Expected: [1, 10])")
    
    # 4. Backward Pass (Dummy loss)
    dummy_loss = y_out.mean()
    dummy_loss.backward()
    
    print("\n--- Structural Sanity Audit ---")
    print(f"Gradients on Input X computed?       {x_sample.grad is not None}")
    print(f"Gradients on Template W computed?    {layer.W.grad is not None}")
    print(f"Gradients on Bias computed?          {layer.bias.grad is not None}")
    
    if x_sample.grad is not None and layer.W.grad is not None:
        print("\n[SUCCESS] Custom Autograd successfully routes through nn.Module parameters.")