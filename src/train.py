# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
from model import ViTBase

class SpecDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files, self.labels = files, labels
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

def train():
    files = glob.glob("/kaggle/working/specs/*/*.png")
    labels = [os.path.basename(os.path.dirname(f)) for f in files]
    label2id = {l: i for i, l in enumerate(sorted(set(labels)))}
    y = [label2id[l] for l in labels]

    train_files, temp_files, train_y, temp_y = train_test_split(files, y, stratify=y, test_size=0.3, random_state=42)
    val_files, _, val_y, _ = train_test_split(temp_files, temp_y, stratify=temp_y, test_size=0.5, random_state=42)

    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    train_ds = SpecDataset(train_files, train_y, transform)
    val_ds = SpecDataset(val_files, val_y, transform)

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16)

    device = torch.device('cuda')
    model = ViTBase(len(label2id)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    for epoch in range(30):
        model.train()
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done")

    torch.save(model.state_dict(), "best_vit.pth")
    return label2id
