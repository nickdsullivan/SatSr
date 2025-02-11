import torch
import torch.nn as nn


class Vae(nn.Module):
    def __init__(self, in_channels, latent_dim, out_channels):
        super(Vae, self).__init__()
        self.encoder = Encoder(in_channels,latent_dim)
        self.decoder = Decoder(latent_dim, out_channels)
    def forward(self, x):
        mu, logvar = self.encoder(x)  
        z = self.reparameterize(mu, logvar)  
        x_reconstructed = self.decoder(z)  
        return x_reconstructed, mu, logvar 
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std 

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(Decoder, self).__init__()
        # Upsample layers, modify according to desired output resolution
        self.fc = nn.Linear(latent_dim, 512*8*8)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
    def forward(self, z):
        x = torch.relu(self.fc(z))
        x = x.view(x.size(0), 512, 8, 8)  # Reshape to feature map size
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))  # Sigmoid for output in range [0, 1]
        return x
    

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Encoder, self).__init__()
        # Add layers to downsample the image (you can adjust kernel sizes, stride, etc.)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        # Final output channels define your latent space dimensionality
        self.fc_mu = nn.Linear(512*8*8, latent_dim)  # e.g., latent space size = latent_dim
        self.fc_logvar = nn.Linear(512*8*8, latent_dim)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten the feature map
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar