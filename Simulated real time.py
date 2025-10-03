import os
import time
import csv
import torch
from torchvision import transforms
from PIL import Image, ImageFile
import cv2
import numpy as np
from google.colab.patches import cv2_imshow 

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("resnet18_trashnet.pt", map_location=device)
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
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()


def conveyor_simulation(image_folder, csv_file="results.csv", interval=2, conf_threshold=0.7):
   
    if not os.path.isdir(image_folder):
        print(f"Error: Image folder not found at '{image_folder}'. Please create this folder and add images, or specify an existing folder.")
        return 

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))]

    if not image_files:
        print(f"No image files found in '{image_folder}'. Please add some .jpg or .png images to the folder.")
        return 


    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Filename", "Predicted_Class", "Confidence", "Low_Confidence_Flag"])

     
        for i, filename in enumerate(sorted(image_files)):
            image_path = os.path.join(image_folder, filename)

            
            label, confidence = predict_image(image_path)
            flag = "⚠️ LOW CONFIDENCE" if confidence < conf_threshold else ""

         
            print(f"Frame {i+1}: {filename} → {label} ({confidence:.2f}) {flag}")

       
            writer.writerow([i+1, filename, label, f"{confidence:.2f}", flag])

            
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            text = f"{label} ({confidence:.2f})"
            color = (0, 255, 0) if confidence >= conf_threshold else (0, 0, 255)

            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            if flag:
                cv2.putText(frame, flag, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2_imshow(frame) 
            time.sleep(interval)  

    print(f"\n✅ Simulation complete. Results saved in {csv_file}")

conveyor_simulation("frames", interval=1, conf_threshold=0.75)
