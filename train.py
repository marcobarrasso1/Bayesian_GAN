from model import * 
import torch 
from datasets import load_dataset
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from utils import CustomDataset
import torch.nn as nn

'''
class BuildDataset(IterableDataset):
    def __init__(self, split="train", transform=None):
        self.dataset = load_dataset("huggan/wikiart", split=split, streaming=True)
        self.transform = transform

    def __iter__(self):
        for sample in self.dataset:
            img = sample["image"]
            genre = sample["genre"]
            if self.transform:
                try:
                    img = self.transform(img)
                except Exception as e:
                    print(f"Skipping corrupted image: {e}")
                    continue
            yield img, genre
'''
    
BATCH_SIZE = 128
LATENT_DIM = 100
FEATURE_MAP = 64
IMG_SIZE = 32
IMG_CHANNELS = 1
NUM_CLASSES = 10
NUM_EPOCHS = 4

logdir = f"./results/"
print(logdir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
writer = SummaryWriter(logdir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

torch.set_float32_matmul_precision('high') 

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    #transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*IMG_CHANNELS, [0.5]*IMG_CHANNELS)
])

'''
dataset = load_dataset("huggan/wikiart", split="train", cache_dir="./wikiart")
genres = [4]
dataset = dataset.filter(lambda example: example["genre"] in genres)
print(len(dataset))
wikidataset = WikiArtDataset(dataset, transform=transform)
dataloader = DataLoader(wikidataset, batch_size=BATCH_SIZE, shuffle=True)
'''

#wikiart_stream = BuildDataset(transform=transform)
#dataloader = DataLoader(wikiart_stream, batch_size=256)

dataset = datasets.MNIST(
    root="./dataset/", 
    transform=transform, 
    train=True,                 
    download=True               
)


dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Train Batches: ", len(dataloader))

G = Generator(LATENT_DIM, NUM_CLASSES, IMG_CHANNELS, FEATURE_MAP, mnist=True).to(device)
D = Discriminator(IMG_CHANNELS, NUM_CLASSES, FEATURE_MAP).to(device)
G.apply(weights_init)
D.apply(weights_init)
G.compile()
D.compile()

load = False
if load:
    G.load_state_dict(torch.load("", map_location=device))
    D.load_state_dict(torch.load("", map_location=device))

g_parameters = sum(p.numel() for p in G.parameters())/1e6
d_parameters = sum(p.numel() for p in D.parameters())/1e6
print(d_parameters + g_parameters, "M parameters")

z = torch.randn(256, LATENT_DIM, device=device)
random_genres = torch.randint(0, NUM_CLASSES, (256,), device=device)

criterion = torch.nn.BCEWithLogitsLoss()
lr = 0.0002
beta1 = 0.5
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

os.makedirs("generated", exist_ok=True)

update = 0
for epoch in range(NUM_EPOCHS):
    prob = 0
    for real_imgs, genre in tqdm(dataloader):
        update += 1
        # --------------------------------
        # 1. Train Discriminator
        # --------------------------------
        optimizer_D.zero_grad()

        real_imgs = real_imgs.to(device)
        genre = genre.to(device)
        batch_size = real_imgs.size(0)

        # Get real output
        real_output = D(real_imgs, genre)
        real_labels = torch.ones_like(real_output, device=device) * 0.9

        real_loss = criterion(real_output, real_labels)

        # Fake images
        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        random_genres = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)
        fake_imgs = G(z, random_genres).detach()  # Detach so gradients don't flow to G
        fake_output = D(fake_imgs, random_genres)
        fake_labels = torch.zeros_like(fake_output, device=device)

        fake_loss = criterion(fake_output, fake_labels)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # --------------------------------
        # 2. Train Generator
        # --------------------------------
        optimizer_G.zero_grad()

        # Generator wants discriminator to output 1 for fakes
        #z = torch.randn(batch_size, latent_dim, device=device)
        #random_genres = torch.randint(0, num_classes, (batch_size,), device=device)
        #z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        fake_imgs = G(z, random_genres)  # regenerate fake images with gradients
        output = D(fake_imgs, random_genres)
        g_labels = torch.ones_like(output, device=device)

        g_loss = criterion(output, g_labels)

        g_loss.backward()
        optimizer_G.step()

        prob += nn.Sigmoid()(output).mean().item()
        writer.add_scalars("Loss", {
        "Discriminator": d_loss.item(),
        "Generator": g_loss.item()
        }, update)
        
        if update % 20 == 0:
            print(f"[Epoch {epoch}] D(x):{torch.sigmoid(real_output).mean().item():.2f} D(G(z)): {nn.Sigmoid()(output).mean().item():.4f}, G_loss: {g_loss.item()}, D_loss: {d_loss.item()}")    
            save_image(fake_imgs[:25], f"generated/sample_{epoch}.png", nrow=5, normalize=True)
            print("Images saved")
            
    torch.save(G.state_dict(), f"./weights/G_parameters{epoch}")
    torch.save(D.state_dict(), f"./weights/D_parameters{epoch}")
    print("Saved model parameters")
        
        

    


