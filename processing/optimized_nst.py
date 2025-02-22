import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from PIL import Image
import numpy as np

class OptimizedNST:
    def __init__(self, device=None):
        self.device = device if device else torch.device('cpu')
        # Load pre-trained VGG19 model with weights
        self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def gram_matrix(self, tensor):
        b, c, h, w = tensor.size()
        tensor = tensor.view(b, c, h * w)
        return torch.bmm(tensor, tensor.transpose(1, 2)) / (c * h * w)

    def get_features(self, x, layers=None):
        if layers is None:
            layers = {
                '0': 'conv1_1',
                '5': 'conv2_1',
                '10': 'conv3_1',
                '19': 'conv4_1',
                '21': 'conv4_2',
                '28': 'conv5_1'
            }
        
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def neural_style_transfer(self, cfg):
        try:
            # Move inputs to device
            content_img = cfg['content_img'].to(self.device)
            style_img = cfg['style_img'].to(self.device)

            # Extract features
            with torch.no_grad():
                content_features = self.get_features(content_img)
                style_features = self.get_features(style_img)
                style_grams = {layer: self.gram_matrix(style_features[layer]) 
                            for layer in style_features}

            # Initialize generated image
            generated = content_img.clone().requires_grad_(True)
            optimizer = Adam([generated], lr=0.02, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

            best_loss = float('inf')
            best_image = None

            # Style transfer loop
            for i in range(cfg['niter']):
                optimizer.zero_grad()
                gen_features = self.get_features(generated)

                # Content loss
                content_loss = F.mse_loss(
                    gen_features['conv4_2'],
                    content_features['conv4_2']
                )

                # Style loss
                style_loss = 0
                for layer in style_features:
                    gen_gram = self.gram_matrix(gen_features[layer])
                    style_loss += F.mse_loss(gen_gram, style_grams[layer]) / len(style_features)

                # Total variation loss
                tv_loss = torch.mean(torch.abs(generated[:, :, :, :-1] - generated[:, :, :, 1:])) + \
                         torch.mean(torch.abs(generated[:, :, :-1, :] - generated[:, :, 1:, :]))

                # Total loss
                loss = cfg['content_weight'] * content_loss + \
                       cfg['style_weight'] * style_loss + \
                       cfg['tv_weight'] * tv_loss

                loss.backward()
                optimizer.step()
                scheduler.step()

                # Save best result
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_image = generated.clone()

                # Print progress
                if (i + 1) % 50 == 0:
                    print(f"Iteration {i+1}/{cfg['niter']}, "
                          f"Content Loss: {content_loss.item():.4f}, "
                          f"Style Loss: {style_loss.item():.4f}, "
                          f"TV Loss: {tv_loss.item():.4f}")

            # Return best result
            return best_image.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)

        except Exception as e:
            print(f"Error in neural_style_transfer: {str(e)}")
            raise


def preprocess_image(image_path, device, image_size=512):
    """Preprocess the image for style transfer."""
    try:
        # Debug: Print the image path
        print(f"\nProcessing image: {image_path}")
        
        image = Image.open(image_path)
        print(f"Original image mode: {image.mode}, size: {image.size}")

        # Convert to RGB even if it claims to be RGB (force standardization)
        if image.mode != 'RGB':
            print(f"Converting from {image.mode} to RGB")
            image = image.convert('RGB')
        else:
            # Ensure no alpha channel is present
            image = image.convert('RGB')  # Redundant but safe

        # Debug: Check image bands after conversion
        print(f"Post-conversion bands: {len(image.getbands())}")

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(image).unsqueeze(0).to(device)
        print(f"Final tensor shape: {tensor.shape}")
        
        # Explicitly ensure 3 channels (slice if necessary)
        if tensor.shape[1] != 3:
            print(f"WARNING: Tensor has {tensor.shape[1]} channels. Slicing to 3 channels.")
            tensor = tensor[:, :3, :, :]  # Take first 3 channels
        
        return tensor
    
    except Exception as e:
        print(f"CRITICAL ERROR in {image_path}: {str(e)}")
        raise


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nst = OptimizedNST(device)

    # Paths to content and style images
    content_path = "path/to/content_image.jpg"
    style_path = "path/to/style_image.jpg"

    # Preprocess the images
    content_img = preprocess_image(content_path, device)
    style_img = preprocess_image(style_path, device)

    # Neural style transfer configuration
    config = {
        'content_img': content_img,
        'style_img': style_img,
        'content_weight': 1e4,
        'style_weight': 1e-2,
        'tv_weight': 1e-6,
        'niter': 400
    }

    # Perform style transfer
    result = nst.neural_style_transfer(config)

    # Convert result to PIL and save
    result_image = Image.fromarray((result * 255).clip(0, 255).astype(np.uint8))
    result_image.save("output_image.jpg")
    print("Style transfer complete! Output saved as 'output_image.jpg'.")
