import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

class SimCLRDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img = self.dataset[idx]  # no unpacking, just the image
        return self.transform(img), self.transform(img)

    def __len__(self):
        return len(self.dataset)

def train_simclr(model, loader, criterion, optimizer=None, epochs=100, device=None):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for x1, x2 in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss / len(loader):.4f}")
