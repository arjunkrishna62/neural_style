import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from .device_manager import DeviceManager

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.conv(x)


class FastNST(nn.Module):
    def __init__(self):
        super(FastNST, self).__init__()
        # Define layers for style transfer (example: pretrained layers, transformer network)
        self.style_transfer_network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.style_transfer_network(x)

    def load_model(self, path):
        """Load pre-trained weights."""
        try:
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {e}")

class FastStyleTransfer:
    def __init__(self, device):
        self.device_manager = DeviceManager()
        self.model = FastNST()
        self.model.load_model("models/fast_nst.pth")  # Load model weights
        self.model = self.model.to(device)  # Move model to the correct device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def stylize(self, content_image):
        """Style transfer inference"""
        try:
            return self.device_manager.inference_step(self.model, content_image)
        except Exception as e:
            self.device_manager.logger.error(f"Error during style transfer: {e}")
            self.device_manager.optimize_memory()
            raise