import numpy as np
from src.data import add_noise
from src.issarkfcm import issarkfcm
from src.metrics import vpc, vpe

def run_system_test():
    print("=== STARTING FULL SYSTEM INTEGRATION TEST ===")
    
    # 1. Create a dummy synthetic brain-slice image slice (64x64 pixels)
    np.random.seed(42)
    mock_brain = np.zeros((64, 64))
    mock_brain[15:35, 15:35] = 0.4  # Simulated Gray Matter cluster
    mock_brain[40:60, 40:60] = 0.8  # Simulated White Matter cluster
    
    # 2. Use your custom data noise module
    mock_noisy_brain = add_noise(mock_brain, level=0.05)
    
    # 3. Fire up the integrated system loop (low max_iter for speed check)
    try:
        U, V = issarkfcm(mock_noisy_brain, c=3, m=2.0, max_iter=3, eps=1e-3)
        print("✔ Complete loop completed without breaking!")
        
        # 4. Score the results using your metrics engine
        score_vpc = vpc(U)
        score_vpe = vpe(U)
        print(f"✔ Metrics validation results:")
        print(f"   - Partition Coefficient (Vpc): {score_vpc:.4f}")
        print(f"   - Partition Entropy (Vpe): {score_vpe:.4f}")
        print("=== INTEGRATION TEST PASSED SUCCESSFULLY ===")
        
    except Exception as e:
        print("❌ CRITICAL ERROR: Pipeline broken down during matrix calculation.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_system_test()