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


class MNCPrototypicalNetwork(nn.Module):
    """
    Non-Parametric Prototypical Readout Head Network.
    Wraps a frozen representation backbone and classifies queries by 
    calculating the L1 distance to online-accumulated class prototype vectors.
    """
    def __init__(self, representation_backbone, bottleneck_dim=32, num_classes=10):
        super().__init__()
        self.backbone = representation_backbone
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        
        # Buffer to store averaged prototype coordinate vectors
        self.register_buffer("prototypes", torch.zeros(num_classes, bottleneck_dim))
        self.register_buffer("prototype_counts", torch.zeros(num_classes))
        
    @torch.no_grad()
    def add_fact(self, x, label):
        """
        Extracts the bottleneck representation of a statement and updates
        the running average prototype vector for the given label.
        """
        self.eval()
        z = self.backbone(x)  # shape: [Batch, bottleneck_dim]
        z_vec = z[0]
        
        count = self.prototype_counts[label].item()
        self.prototypes[label] = (self.prototypes[label] * count + z_vec) / (count + 1)
        self.prototype_counts[label] += 1

    def forward(self, x):
        """
        Computes negative L1 distance from query bottleneck embeddings to
        the stored prototypical class vectors.
        Returns:
            Tensor of shape [Batch, Num_Classes] containing negative L1 distances.
        """
        z = self.backbone(x)  # shape: [Batch, bottleneck_dim]
        
        # Pairwise L1 distance:
        # z: [Batch, 1, bottleneck_dim]
        # prototypes: [1, num_classes, bottleneck_dim]
        diff = z.unsqueeze(1) - self.prototypes.unsqueeze(0)  # [Batch, num_classes, bottleneck_dim]
        l1_dist = diff.abs().sum(dim=2)  # [Batch, num_classes]
        
        # Return negative L1 distance for argmax compatibility
        return -l1_dist


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