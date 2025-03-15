import torch

def gram_matrix(y):
    """
    Calculate Gram matrix for style representation
    
    Args:
        y: feature maps tensor of shape [batch_size, feature_dim, height, width]
    
    Returns:
        Gram matrix tensor
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    """
    Normalize a batch of images using ImageNet mean and std
    
    Args:
        batch: tensor of shape [batch_size, channels, height, width]
    
    Returns:
        Normalized batch tensor
    """
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div(255.0)
    return (batch - mean) / std
