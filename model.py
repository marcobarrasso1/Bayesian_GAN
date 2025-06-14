import torch
import torch.nn as nn

# Generator: Converts noise vector to an image
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_channels, feature_maps, mnist=None):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # Latent x 1 x 1
        self.model = nn.Sequential(
            #nn.ConvTranspose2d(latent_dim + num_classes, feature_maps * 16, 4, 1, 0, bias=False),
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(),
            
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(feature_maps),
            #nn.ReLU(),

            #nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(feature_maps),
            #nn.ReLU(),
            
            nn.Tanh()
        )
        

    def forward(self, noise, labels=None):
        x = noise
        #label_embedding = self.label_emb(labels)  # Shape: [B, num_classes]
        #x = torch.cat([noise, label_embedding], dim=1)  # Shape: [B, latent_dim + num_classes]
        x = x.unsqueeze(2).unsqueeze(3)  # Shape: [B, C, 1, 1]
        return self.model(x)

# Discriminator: Determines if an image is real or fake
class Discriminator(nn.Module):
    def __init__(self, img_channels, num_classes, feature_maps):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            #nn.Conv2d(img_channels + num_classes, feature_maps, 4, 2, 1, bias=False),
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 8, 1, 2, 1, 0, bias=False)
        )
        

    def forward(self, imgs, labels=None):
        x = imgs
        #label_embedding = self.label_emb(labels)  # [B, num_classes]
        #label_map = label_embedding.unsqueeze(2).unsqueeze(3).expand(-1, -1, imgs.size(2), imgs.size(3))  # [B, C, H, W]
        #x = torch.cat([imgs, label_map], dim=1)  # [B, img_channels + label_channels, H, W]
        return self.model(x).view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)