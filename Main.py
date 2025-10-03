from datasets import load_dataset
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

ds = load_dataset("garythung/trashnet")
split_ds = ds["train"].train_test_split(test_size=0.2, seed=42)
train_valid = split_ds["train"]
test_ds = split_ds["test"]

split_valid = train_valid.train_test_split(test_size=0.125, seed=42)
train_ds = split_valid["train"]
valid_ds = split_valid["test"]

print(f"Train size: {len(train_ds)}, Valid size: {len(valid_ds)}, Test size: {len(test_ds)}")

from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def custom_collate_fn(batch):
    images = [transform(item['image']) for item in batch] # Removed Image.fromarray()
    labels = [item['label'] for item in batch]
    return {
        'pixel_values': torch.stack(images),
        'label': torch.tensor(labels)
    }


train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
valid_loader = DataLoader(valid_ds, batch_size=16, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_ds, batch_size=16, collate_fn=custom_collate_fn)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=16)
test_loader = DataLoader(test_ds, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_ds.features["label"].names)

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        images = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        images = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct / total:.4f}")

torch.save(model.state_dict(), "resnet18_trashnet.pth")
print("✅ Model saved as resnet18_trashnet.pth")
