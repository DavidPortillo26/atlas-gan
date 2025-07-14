import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, wandb
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataloader import get_celeba_loader
from torch import nn, optim

def train_gan(epochs=25, batch_size=64, latent_dim=100, data_dir="data/celeba"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="celeba-gan", config={"epochs": epochs, "batch_size": batch_size})

    dataloader = get_celeba_loader(data_dir, batch_size)
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            real = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            gen_imgs = generator(z)

            loss_G = criterion(discriminator(gen_imgs), real)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            loss_real = criterion(discriminator(real_imgs), real)
            loss_fake = criterion(discriminator(gen_imgs.detach()), fake)
            loss_D = (loss_real + loss_fake) / 2
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        wandb.log({"G Loss": loss_G.item(), "D Loss": loss_D.item(), "Generated": [wandb.Image(gen_imgs[0].detach().cpu())]})
        print(f"Epoch {epoch+1}/{epochs} | G: {loss_G.item():.4f} | D: {loss_D.item():.4f}")

if __name__ == "__main__":
    train_gan()
