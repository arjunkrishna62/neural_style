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
    def __init__(self):
        self.eps = 0.05

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
    def protect_style_image(self, style_image, num_steps=100, strength=0.5, progress_callback=None):
        """Process PIL Image directly with Streamlit integration"""
        # Convert PIL to tensor
        style_tensor = self.preprocess_image(style_image)
        
        effective_eps = self.eps * strength * 2  
        
        # Extract style features BEFORE optimization loop
        with torch.no_grad():
            style_features = self.vgg(style_tensor)  # This was missing!
            style_gram = [gram_matrix(f) for f in style_features]
        
        # Initialize perturbation
        pert = torch.zeros_like(style_tensor, requires_grad=True)
        optimizer = optim.Adam([pert], lr=0.01)
        
        # Optimization loop
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Apply perturbation
            perturbed_style = style_tensor + self.eps * strength * torch.tanh(pert)
            perturbed_features = self.vgg(perturbed_style)
            
            # Compute style loss using precomputed style_gram
            perturbed_gram = [gram_matrix(f) for f in perturbed_features]
            style_loss = 0
            for pg, sg in zip(perturbed_gram, style_gram):
                style_loss += torch.mean((pg - sg) ** 2)
            
            (-style_loss).backward()
            optimizer.step()
            
            # Streamlit progress updates
            if progress_callback and step % 10 == 0:
                progress_callback((step+1)/num_steps, f"Step {step+1}/{num_steps}")

        # Final protected image
        with torch.no_grad():
            protected_tensor = style_tensor + effective_eps * torch.tanh(pert)
        
        # Convert to PIL Image for Streamlit
        return self.tensor_to_pil(protected_tensor)

    def tensor_to_pil(self, tensor):
        """Convert protected tensor to PIL Image"""
        tensor = tensor.squeeze(0).cpu()
        tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        tensor = tensor.clamp(0, 1).permute(1, 2, 0).numpy()
        return Image.fromarray((tensor * 255).astype(np.uint8))
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