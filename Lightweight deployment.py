dummy_input = torch.randn(1, 3, 224, 224).to(device)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("resnet18_trashnet.pt")
print("✅ TorchScript model saved as resnet18_trashnet.pt")

from google.colab import files
from PIL import Image, ImageFile
import torch
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


uploaded = files.upload()
image_path = list(uploaded.keys())[0]  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = torch.jit.load("resnet18_trashnet.pt")
model.to(device) 
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")  
    img = transform(img).unsqueeze(0)
    img = img.to(device) 

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()

label, confidence = predict_image(image_path)
print(f"Predicted: {label} ({confidence:.2f})")
