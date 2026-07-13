import os
import sys

# 1. Allow the script to look up one level to find the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.issarkfcm import issarkfcm
# from src.baselines import fcm, kfcm, arkfcm # Add your other baselines when ready

def add_gaussian_noise(image: np.ndarray, noise_level: float) -> np.ndarray:
    """Adds zero-mean Gaussian noise scaled by intensity range."""
    rng = np.random.default_rng(42)
    img_range = image.max() - image.min()
    noise = rng.normal(0, noise_level * img_range, image.shape)
    noisy_image = np.clip(image + noise, 0.0, 1.0)
    return noisy_image

def calculate_metrics(U: np.ndarray) -> tuple[float, float]:
    """Computes Partition Coefficient (Vpc) and Partition Entropy (Vpe)."""
    N, c = U.shape
    Vpc = float(np.sum(U ** 2) / N)
    Vpe = float(-np.sum(U * np.log(U + 1e-10)) / N)
    return Vpc, Vpe

def run_full_suite(clean_slice: np.ndarray):
    # Ensure paths are created relative to the project root
    os.makedirs("outputs/figures", exist_ok=True)
    noise_levels = [0.01, 0.05, 0.09]  # 1%, 5%, 9% noise scales
    results_log = []

    for idx, noise in enumerate(noise_levels):
        print(f"\n=========================================")
        print(f" TESTING NOISE LEVEL: {noise*100}% ")
        print(f"=========================================")
        
        noisy_img = add_gaussian_noise(clean_slice, noise)
        
        # Dictionary to store segmentations for plotting
        saved_segmentations = {"Noisy Image": noisy_img}
        
        # Executing your newly integrated main model
        print("-> Executing ISSARKFCM...")
        U, V = issarkfcm(noisy_img, c=3, m=2.0, max_iter=10, eps=1e-3)
        
        # Generate hard labels for visualization
        labels = np.argmax(U, axis=1).reshape(clean_slice.shape)
        saved_segmentations["ISSARKFCM"] = labels
        
        # Compute quantitative accuracy metrics
        vpc, vpe = calculate_metrics(U)
        results_log.append({
            "Noise Level": f"{noise*100}%",
            "Model": "ISSARKFCM",
            "Vpc (Higher Better)": round(vpc, 4),
            "Vpe (Lower Better)": round(vpe, 4)
        })
            
        # --- Export Plots (Figs 18-20 style) ---
        fig, axes = plt.subplots(1, len(saved_segmentations), figsize=(8, 4))
        for ax_idx, (title, data) in enumerate(saved_segmentations.items()):
            cmap = 'gray' if title == "Noisy Image" else 'jet'
            axes[ax_idx].imshow(data, cmap=cmap)
            axes[ax_idx].set_title(title, fontsize=10)
            axes[ax_idx].axis('off')
            
        plt.tight_layout()
        fig_path = f"outputs/figures/noise_{int(noise*100)}_comparison.png"
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✔ Saved visual comparison to: {fig_path}")

 # --- Save Metrics Table (Tables 2 & 3 style) ---
    df = pd.DataFrame(results_log)
    df.to_csv("outputs/benchmark_metrics.csv", index=False)
    print("\n=== FINAL BENCHMARK PERFORMANCE METRICS ===")
    print(df.to_string(index=False))

    # --- NEW: Generate and Save Performance Line Graphs ---
    # Convert noise strings back to floating point or clean numbers for plotting
    df['Noise_Pct'] = df['Noise Level'].str.replace('%', '').astype(float)
    
    plt.figure(figsize=(10, 4))
    
    # Left subplot: Vpc Trend
    plt.subplot(1, 2, 1)
    plt.plot(df['Noise_Pct'], df['Vpc (Higher Better)'], marker='o', linewidth=2, color='red', label='ISSARKFCM')
    plt.title('Partition Coefficient ($V_{pc}$) vs Noise')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('$V_{pc}$ (Higher is Better)')
    plt.grid(True, linestyle='--')
    plt.legend()
    
    # Right subplot: Vpe Trend
    plt.subplot(1, 2, 2)
    plt.plot(df['Noise_Pct'], df['Vpe (Lower Better)'], marker='s', linewidth=2, color='darkred', label='ISSARKFCM')
    plt.title('Partition Entropy ($V_{pe}$) vs Noise')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('$V_{pe}$ (Lower is Better)')
    plt.grid(True, linestyle='--')
    plt.legend()
    
    plt.tight_layout()
    graph_path = "outputs/figures/metrics_trend_graph.png"
    plt.savefig(graph_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✔ Success! Trend line graphs saved to: {graph_path}")



if __name__ == "__main__":
    print("Generating brain phantom slice structure...")
    X, Y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
    r = np.sqrt(X**2 + Y**2)
    mock_brain = np.zeros((128, 128))
    mock_brain[r < 0.8] = 0.3  # Gray Matter proxy
    mock_brain[r < 0.5] = 0.7  # White Matter proxy
    mock_brain[r < 0.2] = 0.1  # CSF proxy
    
    # THIS LINE TRIPPED IT UP: Make sure this function call is present and indented!
    run_full_suite(mock_brain)