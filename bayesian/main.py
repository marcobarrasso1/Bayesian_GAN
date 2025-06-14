import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import numpy as np
# Device configuration: use GPU if available for faster training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Hyperparameters for Bayesian GAN
z_dim = 100           # Dimensionality of generator input noise
J_g = 1           # number of generator samples to maintain
J_d = 1               # number of discriminator samples (we use 1 as in paper)
batch_size = 64       # batch size for real data and for each generator's fake data
alpha = 0.0001          # SGHMC friction term
etag = 1e-4        # SGHMC step size (learning rate)
etad = 3e-5
M = 1                 # number of SGHMC updates per iteration for each network
num_iterations = 4000 # total training iterations (paper used 5000)
print_interval = 100  # how often to print progress
feature_map = 64

# Set up MNIST data loader (28x28 grayscale images, normalized to [-1,1])
transform = transforms.Compose([
    transforms.Resize((32, 32)),     # scale 28×28 → 32×32
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # now in [–1,1]
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


'''
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Fully connected to 7x7x256 feature map
        self.fc = nn.Linear(z_dim, 256*7*7)
        self.bn0 = nn.BatchNorm2d(256)  # batchnorm for 256 feature maps

        # Transposed conv layers to upscale to 28x28
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 7->14
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)    # 14->28

        # Output activation
        self.out_act = nn.Tanh()

    def forward(self, z):
        # z: tensor of shape (N, z_dim)
        x = self.fc(z)                     # (N, 256*7*7)
        x = x.view(-1, 256, 7, 7)          # reshape to (N,256,7,7)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.deconv1(x)               # (N,128,14,14)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.deconv2(x)               # (N,1,28,28)
        x = self.out_act(x)               # apply Tanh to get output in [-1,1]
        return x
'''

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_maps, mnist=None):
        super().__init__()

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
    
    
'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Conv layers: input is 1x28x28
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)    # 28->14
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 14->7
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 7->4 (approx)
        # After conv3, feature map is about 256x4x4
        self.fc = nn.Linear(256*4*4, 1)  # output a single logit

    def forward(self, x):
        # x: (N,1,28,28) input image (normalized to [-1,1])
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = x.view(x.size(0), -1)        # flatten
        logit = self.fc(x)               # no activation here (logit output)
        return logit
'''

class Discriminator(nn.Module):
    def __init__(self, img_channels, feature_maps):
        super().__init__()

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
    
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

# Instantiate generators and discriminator
generators = [Generator(z_dim, 1, 64).to(device) for _ in range(J_g)]
discriminator = Discriminator(1, 64).to(device)

# Initialize weights from N(0, sqrt(variance)) with variance=10
prior_var = 1.0
'''
for net in generators + [discriminator]:
    for param in net.parameters():
        # Initialize each weight and bias from N(0, sqrt(prior_var))
        nn.init.normal_(param, mean=0.0, std=math.sqrt(prior_var))
'''
for net in generators + [discriminator]:
    net.apply(weights_init_normal)
    
# Set up momentum buffers for SGHMC (same shape as parameters)
# momentum_g[i][k] corresponds to momentum for generators[i] parameter k
momentum_g = [ [torch.zeros_like(p.data).to(device) for p in gen.parameters()] for gen in generators ]
# momentum for discriminator
momentum_d = [ torch.zeros_like(p.data).to(device) for p in discriminator.parameters() ]


# Loss function (binary cross-entropy with logits)
criterion = nn.BCEWithLogitsLoss(reduction='mean')  # using sum to accumulate total gradient

# Labels for real and fake
real_label = 1.0
fake_label = 0.0

# Training loop
iter_count = 0
data_iter = iter(train_loader)
scale = len(train_dataset) / batch_size
max_norm = 1.0

for iteration in tqdm(range(1, num_iterations+1)):
    # Reset data iterator if it's exhausted
    try:
        real_imgs, _ = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        real_imgs, _ = next(data_iter)
    real_imgs = real_imgs.to(device)  # real images batch
    
    
    
    # -------------------------
    # Generator(s) Update (SGHMC for each G_j)
    # -------------------------
    for j, gen in enumerate(generators):
        # Sample a batch of latent vectors (10 * batch_size for Monte Carlo estimate)
        z_batch = torch.randn(1 * batch_size, z_dim, device=device)
        # Freeze D's params for generator update (we won't update D here, just use its output)
        # (No need to set requires_grad=False manually, we'll just not step D and zero D grads.)
        
        # We will perform M SGHMC updates for generator j:
        # Use the same noise batch (z_batch) for all M updates of this generator in this iteration.
        for m_step in range(M):
            # Zero grads for this generator and discriminator
            gen.zero_grad()
            discriminator.zero_grad()
            
            # Forward pass: generate fake images and compute D's output
            fake_imgs = gen(z_batch)                      # shape (10*batch_size, 1, 28, 28)
            fake_logits = discriminator(fake_imgs)        # D's logits on fake images
            # Generator loss: encourage D(fake) to be classified as real (target=1)
            ones = torch.ones(fake_logits.size(0), device=device)
            gen_loss = criterion(fake_logits, ones)
            # Backprop to get gradients for generator parameters
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm)
            # At this point, gen.parameters().grad contains ∂(-logL)/∂θ = -∂logL/∂θ.
            # Add gradient of log-prior: which is +θ/10 (because log prior grad = -θ/10, 
            # and we are computing gradient of -log posterior)
            
            for p in gen.parameters():
                p.grad.data.mul_(-1)
                # Step 2: add ∇ log p(θ) = − θ/σ²
                p.grad.data += (p.data / prior_var) / 1000.
            
            # SGHMC update for generator parameters
            # Update momentum and weights
            
            with torch.no_grad():
                for p_index, p in enumerate(gen.parameters()):
                    # v = (1-alpha)*v + eta*grad + Normal(0, 2*alpha*eta)
                    momentum_g[j][p_index].mul_(1 - alpha)
                    momentum_g[j][p_index].add_(etag * p.grad)  # gradient term
                    # Add noise term:
                    noise = torch.randn_like(p.data) * math.sqrt(2 * alpha * etag)
                    momentum_g[j][p_index].add_(noise)
                    #momentum_g[j][p_index].clamp_(-0.1, 0.1)
                    
                    # Update weights: θ = θ + v
                    p.add_(momentum_g[j][p_index])
            
            
            for p in gen.parameters():
                if p.grad is not None:
                    # Manual SGD step
                    p.data -= etag * p.grad
            # end for each parameter
        # end for M steps
    # end for each generator j
    
    # -------------------------
    # 2. Discriminator Update (SGHMC for D)
    # -------------------------
    # Prepare a batch of real and fake data for D
    # We use one batch of real_imgs (already loaded) and one batch of latent per generator
    z = torch.randn(batch_size, z_dim, device=device)
    # Generate fake images from each generator (list of tensors)
    fake_imgs_all = []
    for gen in generators:
        fake_imgs_all.append(gen(z).detach())
    # Concatenate all fake images and corresponding labels
    fake_imgs_all = torch.cat(fake_imgs_all, dim=0)            # shape: (J_g*batch_size, 1, 28, 28)
    real_labels = torch.full((real_imgs.size(0),), real_label, device=device)
    fake_labels = torch.full((fake_imgs_all.size(0),), fake_label, device=device)
    
    # Perform M SGHMC updates for discriminator
    for m_step in range(M):
        discriminator.zero_grad()
        # Note: we don't zero generator grads here because we won't use them in D update.
        
        # Forward pass: D on real and fake batches
        real_logits = discriminator(real_imgs)
        fake_logits = discriminator(fake_imgs_all)
        # Compute discriminator loss: real->1, fake->0
        real_loss = criterion(real_logits, real_labels)   
        fake_loss = criterion(fake_logits, fake_labels)   
        d_loss = (real_loss + fake_loss)
        # Backprop to get gradients for D's parameters
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm)
        
        for p in discriminator.parameters():
            p.grad.data.mul_(-1)
            # Step 2: add ∇ log p(θ) = − θ/σ²
            p.grad.data += (p.data / prior_var) / 1000.
        
        # SGHMC update for discriminator parameters
        with torch.no_grad():
            for p_index, p in enumerate(discriminator.parameters()):
                v = (1-alpha)*v + eta*grad + noise
                momentum_d[p_index].mul_(1 - alpha)
                momentum_d[p_index].add_(etad * p.grad)
                # noise term
                noise = torch.randn_like(p.data) * math.sqrt(2 * alpha * etad)
                momentum_d[p_index].add_(noise)
                
                #momentum_g[j][p_index].clamp_(-0.1, 0.1)
                # weight update
                p.add_(momentum_d[p_index])
        
        
        for p in discriminator.parameters():
                if p.grad is not None:
                    # Manual SGD step
                    p.data -= etad * p.grad
        # end for each param
    # end for M steps
    
    # (Optionally) print training progress
    if iteration % print_interval == 0:
    # Compute losses per sample
        d_loss_val = d_loss.item() 
        g_loss_val = gen_loss.item() 

        # Compute D(x) and D(G(z)) probabilities
        with torch.no_grad():
            # D(x): run discriminator on the real batch
            real_probs = torch.sigmoid(discriminator(real_imgs))
            d_x = real_probs.mean().item()

            # D(G(z)): use the same fake batch from the last discriminator update
            fake_probs = torch.sigmoid(discriminator(fake_imgs_all))
            d_gz = fake_probs.mean().item()

        print(f"Iter {iteration}/{num_iterations} | "
            f"D_loss: {d_loss_val:.4f} | G_loss: {g_loss_val:.4f} | "
            f"D(x): {d_x:.4f} | D(G(z)): {d_gz:.4f}")


# Choose how many generators & samples per generator to visualize
num_g_to_show = 1   # now 1
imgs_per_gen  = 8

# Random latent vectors (shared across generators for consistency)
vis_z = torch.randn(imgs_per_gen, z_dim, device=device)

# Generate and denormalize images from each generator
gen_images = []
for i in range(num_g_to_show):
    gen = generators[i]
    gen.eval()
    with torch.no_grad():
        imgs = gen(vis_z).cpu()  # shape: (imgs_per_gen, 1, 28, 28)
    gen.train()
    imgs = (imgs * 0.5) + 0.5   # [-1,1] → [0,1]
    gen_images.append(imgs.squeeze(1))  # (imgs_per_gen, 28, 28)

# Create output directory if needed
os.makedirs('outputs', exist_ok=True)

# Make a num_g_to_show × imgs_per_gen grid of subplots
fig, axs = plt.subplots(num_g_to_show, imgs_per_gen,
                       figsize=(2*imgs_per_gen, 2*num_g_to_show))
# reshape so axs is always 2D
axs = np.array(axs).reshape(num_g_to_show, imgs_per_gen)

# Fill in the grid
for row in range(num_g_to_show):
    for col in range(imgs_per_gen):
        axs[row, col].imshow(gen_images[row][col].numpy(), cmap='gray')
        axs[row, col].axis('off')
    axs[row, 0].set_title(f"Generator {row+1}", y=1.02, fontsize=10)

plt.tight_layout()
out_path = os.path.join('outputs', 'bayesgan_styles.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Saved style grid to {out_path}")
