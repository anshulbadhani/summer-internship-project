# src/data.py
import numpy as np
import nibabel as nib

def load_dataset(name: str) -> np.ndarray:
    """
    Loads BrainWeb, IBSR, or a baseline synthetic image.
    For BrainWeb/IBSR, reads NIfTI files using nibabel.
    """
    if name.lower() == "synthetic":
        # Create a clean, pure-NumPy synthetic structural grid with 5 distinct intensity levels
        # (Simulating Background, CSF, Gray Matter, White Matter, and Deep Tissue)
        img = np.zeros((180, 140), dtype=np.float32)
        
        img[15:165, 15:125] = 64.0   # Intensity Level 2
        img[35:145, 30:110] = 128.0  # Intensity Level 3
        img[55:125, 45:95] = 192.0   # Intensity Level 4
        img[75:105, 60:80] = 255.0   # Intensity Level 5
        return img
    else:
        # Expecting a file path to a NIfTI file (.nii / .nii.gz)
        try:
            img = nib.load(name)
            return img.get_fdata().astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load dataset path '{name}': {e}")

def add_noise(image: str | np.ndarray, level: float) -> np.ndarray:
    """
    Injects Gaussian noise into the image at specific levels (e.g., 0.01, 0.05, 0.09).
    """
    if isinstance(image, str):
        img_arr = load_dataset(image)
    else:
        img_arr = image.copy()
        
    img_range = np.max(img_arr) - np.min(img_arr)
    noise = np.random.normal(0, level * img_range, img_arr.shape)
    noisy_image = img_arr + noise
    return np.clip(noisy_image, np.min(img_arr), np.max(img_arr))