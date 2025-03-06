import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 10

class StyleTransferDataset(Dataset):
    def __init__(self, content_dir, style_dir):
        self.content_dataset = datasets.ImageFolder(content_dir, transform=self.transform())
        self.style_dataset = datasets.ImageFolder(style_dir, transform=self.transform())
        
    def transform(self):
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return min(len(self.content_dataset), len(self.style_dataset))
    
    def __getitem__(self, idx):
        content_img, _ = self.content_dataset[idx]
        style_img, _ = self.style_dataset[np.random.randint(0, len(self.style_dataset))]
        return content_img, style_img

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class StyleTransferCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )
        
        # Style transfer blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, content, style):
        # Encode both images
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        
        # AdaIN
        adain_feat = self.adaptive_instance_normalization(content_feat, style_feat)
        
        # Decode
        out = self.res_blocks(adain_feat)
        return self.decoder(out)
    
    def adaptive_instance_normalization(self, content, style):
        content_mean = torch.mean(content, dim=[2,3], keepdim=True)
        content_std = torch.std(content, dim=[2,3], keepdim=True) + 1e-5
        style_mean = torch.mean(style, dim=[2,3], keepdim=True)
        style_std = torch.std(style, dim=[2,3], keepdim=True) + 1e-5
        return style_std * (content - content_mean) / content_std + style_mean

def gram_matrix(features):
    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)

class StyleTransferLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, generated, content, style):
        # Content loss
        content_loss = self.mse(generated, content)
        
        # Style loss
        gen_gram = gram_matrix(generated)
        style_gram = gram_matrix(style)
        style_loss = self.mse(gen_gram, style_gram)
        
        # Total variation loss
        tv_loss = torch.mean(torch.abs(generated[:, :, :, :-1] - generated[:, :, :, 1:])) + \
                 torch.mean(torch.abs(generated[:, :, :-1, :] - generated[:, :, 1:, :]))
        
        return 1.0 * content_loss + 1e6 * style_loss + 1e-3 * tv_loss

def denormalize(tensor):
    """Convert normalized tensors back to PIL images"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.cpu().detach()

def show_images(content, style, generated, epoch, batch_idx):
    """Display training progress"""
    plt.figure(figsize=(15, 5))
    
    # Content Image
    plt.subplot(1, 3, 1)
    plt.imshow(denormalize(content[0]).permute(1, 2, 0))
    plt.title("Content Image")
    plt.axis('off')
    
    # Style Image
    plt.subplot(1, 3, 2)
    plt.imshow(denormalize(style[0]).permute(1, 2, 0))
    plt.title("Style Image")
    plt.axis('off')
    
    # Generated Image
    plt.subplot(1, 3, 3)
    plt.imshow(denormalize(generated[0]).permute(1, 2, 0))
    plt.title(f"Generated\nEpoch {epoch+1} Batch {batch_idx}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def train_model(content_dir, style_dir):
    # Initialize dataset and dataloader
    dataset = StyleTransferDataset(content_dir=content_dir, style_dir=style_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model, optimizer, and loss
    model = StyleTransferCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = StyleTransferLoss()
    
    # Training loop
    loss_history = []
    for epoch in range(EPOCHS):
        for batch_idx, (content_imgs, style_imgs) in enumerate(dataloader):
            content_imgs = content_imgs.to(device)
            style_imgs = style_imgs.to(device)
            
            # Generate stylized images
            generated = model(content_imgs, style_imgs)
            
            # Calculate features
            content_features = model.encoder(content_imgs)
            style_features = model.encoder(style_imgs)
            generated_features = model.encoder(generated)
            
            # Compute loss
            loss = criterion(generated_features, content_features, style_features)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
                show_images(content_imgs, style_imgs, generated, epoch, batch_idx)
                
                # Save model checkpoint
                torch.save(model.state_dict(), f"style_transfer_epoch_{epoch+1}.pth")
    
    return model, loss_history

def style_transfer(content_image, style_image, model_path=None):
    """Perform style transfer on a single image pair"""
    model = StyleTransferCNN().to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        generated = model(content_image.to(device), style_image.to(device))
    
    return generated

if __name__ == "__main__":
    # Example usage
    content_dir = "best-artworks-of-all-time/images"
    style_dir = "image-classification/validation"
    
    # Train the model
    model, loss_history = train_model(content_dir, style_dir)
    
    # Plot final loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title("Final Training Loss Curve")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.show()