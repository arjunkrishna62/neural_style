import torch
import torch.nn as nn
from .utils import gram_matrix

def compute_content_loss(content_features, target_features):
    """
    Compute MSE loss between content features and target features
    
    Args:
        content_features: List of feature maps from VGG for content image
        target_features: List of feature maps from VGG for target image
    
    Returns:
        Content loss (scalar tensor)
    """
    # Use relu_4 (4th feature map) for content representation
    content_loss = nn.MSELoss()(content_features[3], target_features[3])
    return content_loss

def compute_style_loss(style_features, target_features):
    """
    Compute style loss using Gram matrices
    
    Args:
        style_features: List of feature maps from VGG for style image
        target_features: List of feature maps from VGG for target image
    
    Returns:
        Style loss (scalar tensor)
    """
    style_loss = 0.0
    style_weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # Equal weights for all layers
    
    for i in range(len(style_features)):
        style_gram = gram_matrix(style_features[i])
        target_gram = gram_matrix(target_features[i])
        layer_loss = nn.MSELoss()(style_gram, target_gram)
        style_loss += style_weights[i] * layer_loss
    
    return style_loss