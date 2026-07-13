import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchCNN(nn.Module):
    """
    LeNet-style CNN for low-resolution MR image patches.
    Follows Fig. 5 in the paper:
    1. Input layer: n x n patch (single channel).
    2. Conv Layer 1: 6 kernels of size 5 x 5, produces 6 feature maps.
    3. Subsampling Layer 1 (Pooling): 2 x 2 max/average pooling.
    4. Conv Layer 2: 12 kernels of size 5 x 5, produces 12 feature maps.
    5. Subsampling Layer 2 (Pooling): 2 x 2 max/average pooling.
    6. Flatten Layer.
    7. Fully Connected Layer: Maps features to a compact latent representation.
    8. Output Layer: Maps to c cluster probability values (using Softmax).
    """
    def __init__(self, patch_size=17, num_classes=3, latent_dim=84, padding=0):
        super(PatchCNN, self).__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.padding = padding
        
        # Conv Layer 1: 1 input channel (grayscale), 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=self.padding)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2: 6 input channels, 12 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5, padding=self.padding)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dynamically determine the size of the flattened feature maps
        with torch.no_grad():
            dummy = torch.zeros(1, 1, patch_size, patch_size)
            x = self.conv1(dummy)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool2(x)
            self.flat_dim = x.numel()
            
        if self.flat_dim == 0:
            raise ValueError(
                f"Patch size {patch_size} is too small for this CNN architecture without padding. "
                "Please increase patch_size or set padding to 2."
            )
            
        # Fully Connected Layer mapping to a compact latent representation
        self.fc1 = nn.Linear(self.flat_dim, self.latent_dim)
        # Output layer mapping to c cluster membership values
        self.fc2 = nn.Linear(self.latent_dim, self.num_classes)
        
    def forward(self, patch):
        """
        Forward pass of the network.
        
        Parameters:
        - patch: PyTorch tensor or numpy array of shape:
                 (batch_size, 1, patch_size, patch_size)
                 (1, patch_size, patch_size) or (patch_size, patch_size)
                 
        Returns:
        - cluster_probs: PyTorch tensor of shape (batch_size, num_classes) representing
                         the cluster membership probability distribution for each patch.
        """
        # Ensure patch is a float tensor
        if isinstance(patch, np.ndarray):
            patch = torch.from_numpy(patch).float()
            
        # Add dimensions if necessary
        if patch.ndim == 2:
            patch = patch.unsqueeze(0).unsqueeze(0)
        elif patch.ndim == 3:
            patch = patch.unsqueeze(0)
            
        # Move tensor to the same device as the model parameters
        device = next(self.parameters()).device
        patch = patch.to(device)
        
        x = self.conv1(patch)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)
        
    def get_weights(self) -> np.ndarray:
        """
        Extracts all trainable weights and biases, flattens them,
        and returns them as a 1D numpy array.
        """
        params = []
        for p in self.parameters():
            if p.requires_grad:
                params.append(p.detach().cpu().numpy().flatten())
        return np.concatenate(params)
        
    def set_weights(self, weights_np: np.ndarray):
        """
        Loads parameters from a 1D numpy array back into the model's layers.
        
        Parameters:
        - weights_np: 1D numpy array containing the weights and biases.
        """
        device = next(self.parameters()).device
        weights_tensor = torch.tensor(weights_np, dtype=torch.float32, device=device)
        current_idx = 0
        with torch.no_grad():
            for p in self.parameters():
                if p.requires_grad:
                    numel = p.numel()
                    val = weights_tensor[current_idx : current_idx + numel].view_as(p)
                    p.copy_(val)
                    current_idx += numel
