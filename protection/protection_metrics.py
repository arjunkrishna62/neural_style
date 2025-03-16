import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_protection_metrics(original_img, protected_img):
    """Calculate image quality metrics between original and protected images"""
    orig_arr = np.array(original_img.convert("RGB"))
    prot_arr = np.array(protected_img.convert("RGB"))
    
    return {
        'ssim': ssim(orig_arr, prot_arr, multichannel=True, channel_axis=-1),
        'psnr': psnr(orig_arr, prot_arr),
        'max_diff': np.abs(orig_arr.astype(float) - prot_arr.astype(float)).max(),
        'mean_diff': np.mean(np.abs(orig_arr.astype(float) - prot_arr.astype(float))),
        'perc_95': np.percentile(np.abs(orig_arr.astype(float) - prot_arr.astype(float)), 95)
    }