import torch, wandb
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataloader import get_celeba_loader
from torch import nn, optim

def train_gan(epochs-25, batch_szize=64, latent_dim=100, data_dir="data/celeba"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="celeba-gan", config={"epochs": epochs, "batchsize"}):

    dataloader = 