from protection_metrics import calculate_protection_metrics
from protection_visuals import generate_heatmap, generate_histograms


def analyze_protection(original_img, protected_img):
    """Orchestrate full protection analysis"""
    return {
        'metrics': calculate_protection_metrics(original_img, protected_img),
        'heatmap': generate_heatmap(original_img, protected_img),
        'histograms': generate_histograms(original_img, protected_img)
    }