import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(EncoderDecoder, self).__init__()
        
        # Encoder layers
        self.enc_conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_relu1 = nn.ReLU(inplace=True)
        
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_relu2 = nn.ReLU(inplace=True)
        
        self.enc_conv3 = nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(latent_dim)
        self.enc_relu3 = nn.ReLU(inplace=True)
        
        # Decoder layers
        self.dec_conv1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_relu1 = nn.ReLU(inplace=True)
        
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(32)
        self.dec_relu2 = nn.ReLU(inplace=True)
        
        self.dec_conv3 = nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_tanh = nn.Tanh()
    
    def encode(self, x):
        x = self.enc_relu1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_relu2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_relu3(self.enc_bn3(self.enc_conv3(x)))
        return x
    
    def decode(self, x):
        x = self.dec_relu1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_relu2(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_tanh(self.dec_conv3(x))
        return x
    
    def forward(self, x):
        latent = self.encode(x)
        output = self.decode(latent)
        return output