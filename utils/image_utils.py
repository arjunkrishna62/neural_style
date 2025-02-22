from PIL import Image
import torch
from torchvision import transforms

def preprocess_image(image_input, max_size=512, match_shape=None):
    """
    Preprocess an image for style transfer. Handles both file paths and tensors.
    
    Args:
        image_input (PIL.Image or str): Image or file path to preprocess.
        max_size (int): Maximum size of the longer side of the image.
        match_shape (tuple): Shape to resize the image to (height, width).
    
    Returns:
        torch.Tensor: Preprocessed image tensor with shape [1, 3, H, W].
    """
    if isinstance(image_input, torch.Tensor):
        # Ensure correct shape [1, 3, H, W]
        if len(image_input.shape) == 3:
            return image_input.unsqueeze(0)
        elif len(image_input.shape) == 4:
            return image_input
        else:
            raise ValueError(f"Unexpected tensor shape: {image_input.shape}")

    img = Image.open(image_input).convert('RGB')
    if match_shape:
        img = img.resize(match_shape, Image.Resampling.LANCZOS)
    else:
        scale = max_size / max(img.size)
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.Resampling.LANCZOS
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def deprocess_image(tensor):
    tensor = tensor.squeeze(0).clamp(0, 1)  # Clamp values to [0, 1]
    return transforms.ToPILImage()(tensor.cpu())