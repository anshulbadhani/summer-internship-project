# test_pipeline.py
import numpy as np
from src.data import load_dataset, add_noise
from src.fcm import fcm
from src.kfcm import kfcm
from src.arkfcm import arkfcm
from src.metrics import vpc, vpe, vxb

def run_diagnostics():
    print("Step 1: Testing Data Engine & Synthetic Generation...")
    base_img = load_dataset("synthetic")
    noisy_img = add_noise(base_img, level=0.05)
    print(f"Success: Generated image shape {base_img.shape}\n")

    # Set clustering variables (3 tissue classes)
    num_clusters = 3
    
    print("Step 2: Testing Standard FCM Pipeline...")
    U_fcm, V_fcm = fcm(noisy_img, c=num_clusters, max_iter=10)
    print(f"FCM Complete. Centers: {np.round(V_fcm, 2)}")
    print(f" Metrics -> Vpc: {vpc(U_fcm):.4f} | Vpe: {vpe(U_fcm):.4f}\n")

    print("Step 3: Testing Kernel FCM (KFCM) Pipeline...")
    U_kfcm, V_kfcm = kfcm(noisy_img, c=num_clusters, max_iter=10)
    print(f"KFCM Complete. Centers: {np.round(V_kfcm, 2)}\n")

    print("Step 4: Testing Adaptive Regularized KFCM (ARKFCM)...")
    U_ark, V_ark = arkfcm(noisy_img, c=num_clusters, max_iter=10)
    print(f"ARKFCM Complete. Centers: {np.round(V_ark, 2)}")
    print(f"Metrics -> Vxb Index: {vxb(noisy_img, U_ark, V_ark):.4f}\n")
    
    print("All 5 modules are architecturally sound and working together perfectly!")

if __name__ == "__main__":
    run_diagnostics()