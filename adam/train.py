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
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.utils import make_grid

class NoiseLoss(nn.Module):
    def __init__(self, params, scale, observed):
        super().__init__()
        # one buffer per parameter for noise
        self.noises = []
        for param in params:
            noise = 0*param.data.cuda() # will fill with normal at each forward
            self.noises.append(noise)
        self.scale = scale
        self.observed = observed
    '''
    def forward(self, params):
        noise_loss = 0.0
        for noise, var in zip(self.noises, params):
            # This is scale * z^T*v
            # The derivative wrt v will become scale*z
            _noise = noise.normal_(0,1)
            noise_loss = noise_loss + scale*torch.sum(Variable(_noise)*var)
            noise_loss /= self.observed
        return noise_loss
    '''
    def forward(self, params):
        noise_loss = 0.0
        for p in params:
            # out-of-place sampling, same shape/device as p
            z = torch.randn_like(p)
            # accumulate scale * <z, p>
            noise_loss = noise_loss + self.scale * torch.sum(z * p)
        return noise_loss / self.observed
    

class PriorLoss(nn.Module):
    def __init__(self, prior_std, observed):
        super().__init__()
        self.prior_std = prior_std
        self.observed = observed

    def forward(self, params):
        loss = 0.0
        for p in params:
            loss = loss + torch.sum(p * p) / (self.prior_std ** 2)
        return loss / self.observed
    


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
        
        
# Device configuration: use GPU if available for faster training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Hyperparameters for Bayesian GAN
z_dim = 100           # Dimensionality of generator input noise
J_g = 5      # number of generator samples to maintain
J_d = 1               # number of discriminator samples (we use 1 as in paper)
batch_size = 64       # batch size for real data and for each generator's fake data
alpha = 0.0001          # SGHMC friction term
eta_g = 2e-4        # SGHMC step size (learning rate)
eta_d = 2e-4
M = 1                 # number of SGHMC updates per iteration for each network
num_iterations = 4000 # total training iterations (paper used 5000)
print_interval = 100  # how often to print progress
feature_map = 64
noise_std_g = math.sqrt(2 * alpha * eta_g)
noise_std_d = math.sqrt(2 * alpha * eta_d)

torch.set_float32_matmul_precision('high') 

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    #transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*1, [0.5]*1)
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

N = len(train_dataset)
print(f"dataset length {N}")


generators = [Generator(z_dim, 1, 64).to(device) for _ in range(J_g)]
D = Discriminator(1, 64).to(device)

#print(len[generators])
optGs = [
    torch.optim.Adam(gen.parameters(), lr=eta_g, betas=(0.5,0.999))
    for gen in generators
]
optD = torch.optim.Adam(D.parameters(), lr=eta_d, betas=(0.5,0.999))

scale = len(train_dataset) / batch_size

noise_G = NoiseLoss(generators[0].parameters(), scale=math.sqrt(2*alpha*eta_g), observed=1000.) 
prior_G = PriorLoss(prior_std=10.0, observed=1000.) 
noise_D = NoiseLoss(D.parameters(), scale=math.sqrt(2*alpha*eta_g), observed=1000.)
prior_D = PriorLoss(prior_std=10.0, observed=1000.)

criterion = nn.BCEWithLogitsLoss()


for net in generators + [D]:
    net.apply(weights_init_normal)
    
generators = [torch.compile(g) for g in generators]
D = torch.compile(D)
    
real_label = 1.0
fake_label = 0.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fixed_noise = torch.randn(16, z_dim, device=device)
NUM_EPOCHS = 5

'''
for iteration in tqdm(range(1, num_iterations+1)):
    # Reset data iterator if it's exhausted
    try:
        real_imgs, _ = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        real_imgs, _ = next(data_iter)
    real_imgs = real_imgs.to(device)
'''

iteration = 0
for epoch in range(NUM_EPOCHS):
    for real_imgs, genre in tqdm(train_loader):
        
        real_imgs = real_imgs.to(device)
        
        optD.zero_grad()
        # Generate fake images from each generator (list of tensors)
        fake_imgs_all = []
        z = torch.randn(batch_size, z_dim, device=device)
        for gen in generators:
            #z = torch.randn(batch_size, z_dim, device=device)
            fake_imgs_all.append(gen(z))
        # Concatenate all fake images and corresponding labels
        fake_imgs_all = torch.cat(fake_imgs_all, dim=0)            # shape: (J_g*batch_size, 1, 28, 28)

        real_logits = D(real_imgs)
        real_labels = torch.ones_like(real_logits, device=device)
        
        fake_logits = D(fake_imgs_all.detach())
        fake_labels = torch.zeros_like(fake_logits, device=device)
        
        real_loss = criterion(real_logits, real_labels)   # sum of -log(D(real))
        fake_loss = criterion(fake_logits, fake_labels) # sum of -log(1-D(fake)) over all fakes
        d_loss = real_loss + fake_loss
        d_loss = real_loss + fake_loss + prior_D(D.parameters()) + noise_D(D.parameters())
        
        d_loss.backward()
        optD.step()
        
        for j, gen in enumerate(generators):
            # Sample a batch of latent vectors (10 * batch_size for Monte Carlo estimate)
            #z_batch = torch.randn(1 * batch_size, z_dim, device=device)
            # Freeze D's params for generator update (we won't update D here, just use its output)
            # (No need to set requires_grad=False manually, we'll just not step D and zero D grads.)
            
            optGs[j].zero_grad()
            '''
            # Forward pass: generate fake images and compute D's output
            z = torch.randn(batch_size, z_dim, device=device)
            fake_imgs = gen(z)             # NO .detach() here!
            #logits = D(fake_j)
            logits = D(fake_imgs)
            #fake_logits = D(fake_imgs_all)        # D's logits on fake images
            # Generator loss: encourage D(fake) to be classified as real (target=1)
            ones = torch.ones_like(logits, device=device)
            gen_loss = criterion(logits, ones)
            #gen_loss = gen_loss + prior_G[j]() + noise_G[j](gen.parameters()) 

            gen_loss.backward()
            optGs[j].step()
            '''
        fake_imgs_all = []
        z = torch.randn(batch_size, z_dim, device=device)
        for gen in generators:
            #z = torch.randn(batch_size, z_dim, device=device)
            fake_imgs_all.append(gen(z))
        fake_imgs_all = torch.cat(fake_imgs_all, dim=0) 
                
        output = D(fake_imgs_all)
        ones = torch.ones_like(output, device=device)
        gen_loss = criterion(output, ones)
        
        for netG in generators:
            gen_loss += prior_G(netG.parameters())
            gen_loss += noise_G(netG.parameters())
        
        gen_loss = gen_loss / J_g
        
        gen_loss.backward()
        for optimizerG in optGs:
            optimizerG.step()
        
        if iteration % print_interval == 0:
        # Compute losses per sample
            d_loss_val = d_loss.item() 
            g_loss_val = gen_loss.item() 

            # Compute D(x) and D(G(z)) probabilities
            with torch.no_grad():
                # D(x): run discriminator on the real batch
                real_probs = torch.sigmoid(real_logits)
                d_x = real_probs.mean().item()

                # D(G(z)): use the same fake batch from the last discriminator update
                fake_probs = torch.sigmoid(fake_logits)
                d_gz = fake_probs.mean().item()

            print(f"Iter {iteration}/{num_iterations} | "
                f"D_loss: {d_loss_val:.4f} | G_loss: {g_loss_val:.4f} | "
                f"D(x): {d_x:.4f} | D(G(z)): {d_gz:.4f}")
            
        iteration += 1
        
    J_g = len(generators)
    imgs_per_gen = 16
    grid_r, grid_c = 4, 4  # each generator’s grid is 4×4

    fig, axes = plt.subplots(
        nrows=grid_r,
        ncols=grid_c * J_g,
        figsize=(2 * grid_c * J_g, 2 * grid_r),
        squeeze=False
    )

    for i, G in enumerate(generators):
        with torch.no_grad():
            imgs = G(fixed_noise.to(device))           # (imgs_per_gen, 1, H, W)
        for idx in range(imgs_per_gen):
            row = idx // grid_c
            col = i * grid_c + (idx % grid_c)
            ax = axes[row][col]
            ax.imshow(imgs[idx].cpu().squeeze(), cmap='gray')
            ax.axis('off')
        # label the top-left cell of this block
        axes[0][i*grid_c].set_title(f"Generator {i+1}", fontsize=12)

    plt.tight_layout(pad=0.5)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f"outputs/epoch{epoch}_all_generators.png", dpi=150)
    plt.close(fig)

'''
vis_z = torch.randn(25, z_dim, device=device)
imgs = gen(vis_z)
save_image(imgs, f"outputs/sample.png", nrow=5, normalize=True)
'''

assert False
num_g_to_show = 1   # now 1
imgs_per_gen  = 25


# Generate and denormalize images from each generator
for i in range(J_g):
    gen = generators[i]
    imgs = gen(vis_z)  # shape: (imgs_per_gen, 1, 28, 28)
    save_image(imgs, f"outputs/gen{i}.png", nrow=5, normalize=True)


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
        axs[row, col].imshow(gen_images[row][col].detach().numpy(), cmap='gray')
        axs[row, col].axis('off')
    axs[row, 0].set_title(f"Generator {row+1}", y=1.02, fontsize=10)

plt.tight_layout()
out_path = os.path.join('outputs', 'bayesgan_styles.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Saved style grid to {out_path}")

