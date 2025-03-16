import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generate_heatmap(original_img, protected_img):
    """Generate difference heatmap visualization"""
    orig_arr = np.array(original_img.convert("RGB"))
    prot_arr = np.array(protected_img.convert("RGB"))
    
    diff = cv2.absdiff(orig_arr, prot_arr)
    heatmap = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

def generate_histograms(original_img, protected_img):
    """Generate comparative histograms of original vs protected images"""
    # Convert images to numpy arrays
    orig_arr = np.array(original_img.convert("RGB"))
    prot_arr = np.array(protected_img.convert("RGB"))
    
    # Create figure with subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original image histogram
    ax[0].hist(orig_arr.ravel(), bins=50, color='blue', alpha=0.5)
    ax[0].set_title('Original Histogram')
    ax[0].set_xlabel('Pixel Value')
    ax[0].set_ylabel('Frequency')
    
    # Protected image histogram
    ax[1].hist(prot_arr.ravel(), bins=50, color='red', alpha=0.5)
    ax[1].set_title('Protected Histogram')
    ax[1].set_xlabel('Pixel Value')
    ax[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig