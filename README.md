# End-to-End-ML-Pipeline-for-Real-Time-Material-Classification
🏭 Scrap Material Classification – End-to-End ML Pipeline
📌 Project Overview

This project implements a real-time scrap material classification pipeline.
Using the TrashNet dataset, we train a CNN-based model, deploy it in a lightweight format (TorchScript/ONNX), and simulate a conveyor belt system where frames are classified at intervals.

The goal is to mimic real-world scenarios of automated scrap sorting while keeping the pipeline lightweight and deployment-ready.

📊 Dataset
Dataset Used: TrashNet
Classes: Cardboard, Glass, Metal, Paper, Plastic, Trash
Chosen because it provides diverse material categories commonly found in waste sorting pipelines.
Preprocessing:
Resized images → 224x224
Normalized with ImageNet mean/std
Train/Validation/Test split = 70/10/20

🧠 Model Development
Architecture: ResNet18 (Transfer Learning)
Training Setup:
Optimizer: Adam (lr=1e-4)
Loss: CrossEntropy
Epochs: 5
Batch size: 16
Evaluation Metrics:
Accuracy
Precision, Recall, F1-score
Confusion Matrix

🚀 Lightweight Deployment
Converted trained model into:
✅ TorchScript (.pt)
✅ ONNX (.onnx)
Inference Script Features:
Takes one image/frame as input
Outputs Predicted class + Confidence score
Works with .jpg and .png

🔄 Conveyor Belt Simulation
A dummy real-time simulation loop that mimics conveyor belt cameras.
Loads frames from a folder at intervals
For each frame:
Classify
Log output → console + save into results.csv
Raise ⚠️ flag if confidence < threshold
Extended with visualization:
Displays each frame with label + confidence overlay
Saves annotated frames in /results/

📈 Performance Summary
Test Accuracy: ~85% (ResNet18 baseline)
Macro F1-score: ~0.83
Confusion matrix & precision/recall per class are included in performance_report.md.
