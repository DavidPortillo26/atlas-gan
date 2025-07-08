# utils/train_baseline.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from models.generator import Generator
from models.discriminator import Discriminator
from utils.mnist_loader import get_mnist_loader

def train_dcgan(epochs=25, batch_size=64, latent_dim=100, device='cuda'):
    wandb.init(project="dcgan-mnist", config={"epochs": epochs, "batch_size": batch_size})
    dataloader = get_mnist_loader(batch_size)
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Labels
            real = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # Train Generator
            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            gen_imgs = generator(z)
            loss_G = criterion(discriminator(gen_imgs), real)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            loss_real = criterion(discriminator(real_imgs), real)
            loss_fake = criterion(discriminator(gen_imgs.detach()), fake)
            loss_D = (loss_real + loss_fake) / 2
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        wandb.log({"Generator Loss": loss_G.item(), "Discriminator Loss": loss_D.item()})
        print(f"Epoch {epoch+1}/{epochs} | G Loss: {loss_G.item():.4f} | D Loss: {loss_D.item():.4f}")

        # Log sample images
        wandb.log({"Generated Images": [wandb.Image(gen_imgs[0].detach().cpu())]})

if __name__ == "__main__":
    train_dcgan()
