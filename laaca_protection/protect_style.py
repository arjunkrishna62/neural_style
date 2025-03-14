import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Import from LAACA files
from .encoder_decoder import EncoderDecoder
from .vgg import VGG
from .utils import gram_matrix, normalize_batch
from .train_eps import compute_content_loss, compute_style_loss

class StyleProtector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.vgg = VGG().to(device)
        self.encoder_decoder = EncoderDecoder().to(device)
        
        # Preprocessing for images
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])
        
        # Loss weights
        self.style_weight = 1e5
        self.content_weight = 1e0
        self.eps = 0.05  # Epsilon for perturbation strength
        
    def preprocess_image(self, image):
        """Convert PIL image to tensor and normalize."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        if isinstance(image, Image.Image):
            # Resize image if needed
            if min(image.size) > 512:
                ratio = 512.0 / min(image.size)
                new_size = (int(ratio * image.size[0]), int(ratio * image.size[1]))
                image = image.resize(new_size, Image.LANCZOS)
            
            image = self.trans(image).unsqueeze(0).to(self.device)
        
        return image
    
    def protect_style_image(self, style_image, num_steps=300, return_tensor=False):
        """
        Apply LAACA protection to a style image.
        
        Args:
            style_image: PIL Image or numpy array or tensor
            num_steps: Number of optimization steps
            return_tensor: If True, return torch tensor, otherwise return numpy array
            
        Returns:
            Protected style image
        """
        # Preprocess style image
        if not torch.is_tensor(style_image):
            style_tensor = self.preprocess_image(style_image)
        else:
            style_tensor = style_image.to(self.device)
        
        # Extract style features
        style_features = self.vgg(style_tensor)
        style_gram = [gram_matrix(f) for f in style_features]
        
        # Initialize random noise for perturbation
        pert = torch.zeros_like(style_tensor, requires_grad=True)
        optimizer = optim.Adam([pert], lr=0.01)
        
        # Optimization loop to find adversarial perturbation
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Apply perturbation
            perturbed_style = style_tensor + self.eps * torch.tanh(pert)
            perturbed_features = self.vgg(perturbed_style)
            
            # Compute style loss
            perturbed_gram = [gram_matrix(f) for f in perturbed_features]
            style_loss = 0
            for pg, sg in zip(perturbed_gram, style_gram):
                style_loss += torch.mean((pg - sg) ** 2)
            
            # We want to maximize the style loss (find adversarial example)
            loss = -style_loss
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f"Step {step}/{num_steps}, Loss: {loss.item()}")
        
        # Final perturbed style image
        with torch.no_grad():
            protected_style = style_tensor + self.eps * torch.tanh(pert)
        
        # Convert to displayable format
        if return_tensor:
            return protected_style
        
        # Denormalize and convert to numpy
        protected_style = protected_style[0].cpu()
        protected_style = protected_style * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                         torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        protected_style = protected_style.clamp(0, 1).numpy()
        protected_style = np.transpose(protected_style, (1, 2, 0)) * 255.0
        
        return protected_style.astype(np.uint8)
    
    def save_protected_image(self, style_image, output_path):
        """Save protected style image to disk."""
        protected = self.protect_style_image(style_image, return_tensor=True)
        
        # Denormalize
        protected = protected[0].cpu()
        protected = protected * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                   torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        protected = protected.clamp(0, 1)
        
        save_image(protected, output_path)
        return output_path
    
    def compare_original_and_protected(self, style_image):
        """Return both original and protected versions for comparison."""
        if not torch.is_tensor(style_image):
            style_tensor = self.preprocess_image(style_image)
        else:
            style_tensor = style_image.to(self.device)
            
        protected_tensor = self.protect_style_image(style_tensor, return_tensor=True)
        
        # Convert both to numpy arrays
        original = style_tensor[0].cpu()
        original = original * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                  torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        original = original.clamp(0, 1).numpy()
        original = np.transpose(original, (1, 2, 0)) * 255.0
        
        protected = protected_tensor[0].cpu()
        protected = protected * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                   torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        protected = protected.clamp(0, 1).numpy()
        protected = np.transpose(protected, (1, 2, 0)) * 255.0
        
        return original.astype(np.uint8), protected.astype(np.uint8)