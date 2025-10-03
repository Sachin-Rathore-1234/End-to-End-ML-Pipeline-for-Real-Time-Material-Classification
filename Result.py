from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def custom_collate_fn(batch):
    images = [transform(item['image']) for item in batch]
    labels = [item['label'] for item in batch]
    return {
        'pixel_values': torch.stack(images),
        'label': torch.tensor(labels)
    }

test_loader = DataLoader(test_ds, batch_size=16, collate_fn=custom_collate_fn)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        images = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=train_ds.features["label"].names))

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)
