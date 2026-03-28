import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
from src.segmentation import segment_lung

st.title("Lung Cancer Detection System")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Get class order directly from training dataset
dataset = datasets.ImageFolder("dataset/train")
classes = dataset.classes  # e.g. ['cancer', 'no_cancer']

# Load trained model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/best_lung_cancer_model.pth", map_location="cpu"))
model.eval()

# Grad-CAM setup
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Upload image
uploaded_file = st.file_uploader("Upload a CT scan image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded CT Scan", use_container_width=True)

    img = transform(image).unsqueeze(0)

    # Model prediction
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    raw_prediction = classes[predicted.item()]
    confidence_score = confidence.item() * 100

    if raw_prediction == "no_cancer":
        prediction = "No Cancer"
        stage = "No Cancer Detected"
    else:
        prediction = "Cancer"
        stage = "Early Stage"

    st.write(f"### Prediction: {prediction}")
    st.write(f"### Confidence Score: {confidence_score:.2f}%")
    st.write(f"### Estimated Stage Group: {stage}")
    st.write("Medical Disclaimer: This system is for research purposes only and not for clinical diagnosis.")

    # Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=img)[0]

    image_np = cv2.resize(np.array(image), (224, 224)) / 255.0

    visualization = show_cam_on_image(
        image_np.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    st.image(visualization, caption="AI Attention Heatmap", use_container_width=True)