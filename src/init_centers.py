import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view

from src.cnn import PatchCNN
from src.de import de_optimize

def run_simple_fcm(intensities, c, m=2.0, max_iter=20, eps=1e-5):
    """
    Standard Fuzzy C-Means (FCM) clustering on 1D pixel intensities.
    Used for pre-clustering to obtain pseudo-labels for CNN training.
    
    Parameters:
    - intensities: 1D numpy array of shape (N,)
    - c: number of clusters.
    - m: fuzzification coefficient (default: 2.0).
    - max_iter: maximum iterations (default: 20).
    - eps: convergence threshold (default: 1e-5).
    
    Returns:
    - labels: 1D numpy array of shape (N,) containing hard cluster labels.
    - centers: 1D numpy array of shape (c,) containing sorted cluster centers.
    """
    N = len(intensities)
    if N == 0:
        raise ValueError("Empty intensity array provided for FCM.")
        
    # Initialize membership matrix U randomly and normalize along rows
    U = np.random.rand(N, c)
    U = U / (np.sum(U, axis=1, keepdims=True) + 1e-15)
    
    for i in range(max_iter):
        U_prev = U.copy()
        
        # Calculate cluster centers
        Um = U ** m
        centers = np.sum(Um * intensities[:, None], axis=0) / (np.sum(Um, axis=0) + 1e-15)
        # Sort centers to ensure stable assignment mapping
        centers = np.sort(centers)
        
        # Calculate distances to centers
        dists = np.abs(intensities[:, None] - centers[None, :])
        dists = np.clip(dists, 1e-10, None)
        
        # Update membership matrix U
        inv_dists = 1.0 / dists
        power = 2.0 / (m - 1)
        inv_dists_power = inv_dists ** power
        U = inv_dists_power / (np.sum(inv_dists_power, axis=1, keepdims=True) + 1e-15)
        
        # Convergence check
        if np.linalg.norm(U - U_prev) < eps:
            break
            
    # Final computation of sorted centers
    Um = U ** m
    centers = np.sum(Um * intensities[:, None], axis=0) / (np.sum(Um, axis=0) + 1e-15)
    sorted_idx = np.argsort(centers)
    centers = centers[sorted_idx]
    U = U[:, sorted_idx]
    
    # Assign hard labels based on argmax of membership
    labels = np.argmax(U, axis=1)
    return labels, centers

def extract_all_patches(image, patch_size):
    """
    Extracts n x n 2D local patches centered at each pixel/voxel.
    Supports 2D images (H, W) and 3D volumes (H, W, D).
    Uses reflection padding at boundaries to keep dimensions consistent.
    
    Parameters:
    - image: 2D or 3D numpy array.
    - patch_size: odd integer representing the size of the patch.
    
    Returns:
    - patches: numpy array of shape (N, 1, patch_size, patch_size) where N = H * W (* D).
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer.")
        
    pad = patch_size // 2
    
    if image.ndim == 2:
        H, W = image.shape
        padded = np.pad(image, pad, mode='reflect')
        patches = sliding_window_view(padded, (patch_size, patch_size))
        # Reshape to (N, 1, patch_size, patch_size) where N = H * W
        return patches.reshape(-1, 1, patch_size, patch_size)
        
    elif image.ndim == 3:
        H, W, D = image.shape
        all_patches = []
        for z in range(D):
            slice_img = image[:, :, z]
            padded = np.pad(slice_img, pad, mode='reflect')
            patches = sliding_window_view(padded, (patch_size, patch_size))
            all_patches.append(patches.reshape(-1, 1, patch_size, patch_size))
        # Concatenate along the voxel/pixel dimension
        return np.concatenate(all_patches, axis=0)
        
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}. Must be 2D or 3D.")

def init_centers_cnn_de(
    image,
    c,
    patch_size=17,
    num_samples=1000,
    m=2.0,
    latent_dim=84,
    verbose=True
) -> np.ndarray:
    """
    Initializes cluster centers using a combined DE-CNN framework.
    
    Workflow:
    1. Pre-cluster the pixel intensities using standard Fuzzy C-Means to get pseudo-labels.
    2. Extract local n x n patches for each pixel/voxel.
    3. Normalize intensities (Z-score normalization).
    4. Randomly sample a subset of patches for the DE loop to accelerate convergence.
    5. Define a fitness function: Cross-Entropy loss between the CNN predictions and pre-clustering labels.
    6. Optimize the CNN weights/biases using the Differential Evolution loop.
    7. Load the best parameters back into the CNN.
    8. Compute the cluster probabilities for all pixels in the image using the optimized CNN.
    9. Compute the initial cluster centers V0 as a membership-weighted average of original intensities.
    10. Sort V0 to ensure consistent tissue ordering (e.g. CSF, GM, WM ordered by intensity).
    
    Parameters:
    - image: 2D or 3D numpy array representing the brain MR image.
    - c: int, number of target clusters (classes).
    - patch_size: int, size of local patch (default: 17).
    - num_samples: int, number of patches to sample for the DE loop (default: 1000).
    - m: float, fuzziness coefficient (default: 2.0).
    - latent_dim: int, compact latent representation size in PatchCNN (default: 84).
    - verbose: bool, print progress messages.
    
    Returns:
    - V0: numpy array of shape (c, feature_dim) where feature_dim=1.
          Represents the optimized initial cluster centers.
    """
    # Robustness check: squeeze image dimensions if it has singleton dimensions (e.g. (H, W, 1) or (1, H, W))
    if image.ndim == 3:
        if image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
        elif image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
            
    intensities = image.flatten()
    N = len(intensities)
    
    if N == 0:
        raise ValueError("Input image is empty.")
        
    # Auto-detect device for PyTorch operations (supports CUDA/MPS/CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    if verbose:
        print(f"Initializing cluster centers using DE-CNN on device: {device}...")
        print("Step 1: Running pre-clustering (FCM)...")
        
    # Step 1: Run pre-clustering to get pseudo-ground-truth labels
    fcm_labels, fcm_centers = run_simple_fcm(intensities, c, m=m)
    
    # Step 2: Normalize image intensities (Z-score normalization)
    img_mean = np.mean(image)
    img_std = np.std(image) + 1e-8
    norm_image = (image - img_mean) / img_std
    
    # Step 3: Extract local image patches
    if verbose:
        print(f"Step 2: Extracting local {patch_size}x{patch_size} patches...")
    patches = extract_all_patches(norm_image, patch_size)
    
    # Step 4: Sample a subset of patches to accelerate the DE loop
    if num_samples is not None and num_samples < N:
        if verbose:
            print(f"Step 3: Sampling {num_samples} patches for the DE loop...")
        np.random.seed(42)
        indices = np.random.choice(N, size=num_samples, replace=False)
        sample_patches = patches[indices]
        sample_labels = fcm_labels[indices]
    else:
        sample_patches = patches
        sample_labels = fcm_labels
        
    # Convert numpy patches and labels to PyTorch tensors and send to the active device
    patches_t = torch.tensor(sample_patches, dtype=torch.float32).to(device)
    labels_t = torch.tensor(sample_labels, dtype=torch.long).to(device)
    one_hot_t = F.one_hot(labels_t, num_classes=c).float().to(device)
    
    # Step 5: Initialize PatchCNN model and move to device
    padding = 2 if patch_size < 17 else 0
    model = PatchCNN(patch_size=patch_size, num_classes=c, latent_dim=latent_dim, padding=padding).to(device)
    
    # Find the dimensionality D of the search space (total weights & biases of the CNN)
    dummy_params = model.get_weights()
    D = len(dummy_params)
    
    # Set the parameter bounds for weights (typically in [-1.0, 1.0])
    low_bound = -1.0 * np.ones(D)
    high_bound = 1.0 * np.ones(D)
    bounds = (low_bound, high_bound)
    
    # Step 6: Define the fitness function (Cross-Entropy loss)
    # We want to find CNN weights that minimize classification error on the pseudo-labels
    def fitness_fn(weights):
        model.set_weights(weights)
        with torch.no_grad():
            probs = model(patches_t)
            # Add epsilon to prevent log(0)
            loss = -torch.mean(torch.sum(one_hot_t * torch.log(probs + 1e-15), dim=1))
            return loss.item()
            
    # Step 7: Run DE optimization
    if verbose:
        print(f"Step 4: Running Differential Evolution (optimizing {D} parameters)...")
    best_weights = de_optimize(
        fitness_fn,
        bounds,
        pop_size=12,
        F=0.8,
        CR=0.7,
        maxiter=300,
        nerp=10,
        eps=1e-6,
        verbose=verbose
    )
    
    # Step 8: Load the best weights into the CNN model
    model.set_weights(best_weights)
    
    # Step 9: Predict membership probabilities for all pixels in batches to prevent memory issues
    if verbose:
        print("Step 5: Computing final cluster memberships with the optimized CNN...")
    all_probs = []
    batch_size = 2048
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_patches = torch.tensor(patches[i:i+batch_size], dtype=torch.float32).to(device)
            probs = model(batch_patches)
            all_probs.append(probs.cpu().numpy())
    U = np.concatenate(all_probs, axis=0) # shape (N, c)
    
    # Step 10: Compute the initial cluster centers V0 on the original intensity scale
    # V0_k = sum_i (u_ik^m * x_i) / sum_i (u_ik^m)
    # If a cluster is empty (or has almost zero membership in U), fall back to initial FCM center
    Um = U ** m
    sums = np.sum(Um, axis=0)
    V0 = np.zeros(c)
    for k in range(c):
        if sums[k] > 1e-5:
            V0[k] = np.sum(Um[:, k] * intensities) / sums[k]
        else:
            V0[k] = fcm_centers[k]
            
    # Sort centers to ensure stable index assignment (by intensity)
    V0 = np.sort(V0)
    
    if verbose:
        print(f"Initial cluster centers V0: {V0}")
        
    # Return shape (c, 1) to match (c, feature_dim) where feature_dim=1
    return V0.reshape(c, 1)

# Alias for backward compatibility
init_centers_cnn_pso = init_centers_cnn_de

