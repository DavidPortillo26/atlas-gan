from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_celeba_loader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
