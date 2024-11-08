import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models
import os
from PIL import Image
import numpy as np


# 1. Rasmni o'qish va transformatsiya qilish
class CustomDataset(Dataset):
    def __init__(self, papka_yoli, transform=None):
        self.papka_yoli = papka_yoli
        self.transform = transform
        self.images = []
        self.labels = []

        # Papkalarni tekshirish
        for label, folder in enumerate(['kasalopka', 'soglomopka']):
            folder_path = os.path.join(papka_yoli, folder)
            for filename in os.listdir(folder_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(folder_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label


# 2. Transformatsiyalar (rasmlarni standartlashtirish)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Ma'lumotlarni yuklash
dataset = CustomDataset('D:/', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. Modelni yaratish (ResNet-18)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 ta sinf (kasalopka va soglomopka)
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 5. Yo'qotish funktsiyasi va optimizatsiya
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 6. Modelni o'qitish
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


# 7. Modelni test qilish
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")


# 8. Modelni o'qitish va test qilish
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, train_loader, criterion, optimizer, num_epochs=5)
test_model(model, test_loader)
