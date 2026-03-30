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
from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")

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
    
    # Segmentation
    lung_mask = segment_lung(image)

    # Prepare image for model
    img = transform(image).unsqueeze(0)

    # Model prediction
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    prediction = classes[predicted.item()]
    confidence_score = confidence.item() * 100

    # Show prediction
    st.write(f"### Prediction: {prediction}")
    st.write(f"### Confidence Score: {confidence_score:.2f}%")

    #Stage estimation
    
    if prediction == "cancer":
        if confidence_score > 85:
          stage = "Early Stage"
        else:
            stage = "Advanced Stage"

        st.write(f"### Estimated Stage Group: {stage}")
    else:
        st.success("No cancer detected")

    # Medical disclaimer
    st.info("⚠️ This system is for educational purposes only and should not be used for medical diagnosis.")

    # Show YOLO-style box only if cancer is predicted
    if prediction == "cancer":
        img_with_boxes = np.array(image).copy()
        h, w, _ = img_with_boxes.shape

        # Demo inner box
        x1 = int(w * 0.30)
        y1 = int(h * 0.25)
        x2 = int(w * 0.70)
        y2 = int(h * 0.65)

        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        st.image(img_with_boxes, caption="YOLO-style Detection (Demo)", use_container_width=True)
    else:
        st.info("No suspicious region displayed because the scan was predicted as no cancer.")

    # Show segmentation
    st.image(lung_mask, caption="Segmented Lung Region", use_container_width=True)

    # Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=img)[0]
    image_np = np.array(image.resize((224, 224))) / 255.0

    visualization = show_cam_on_image(
        image_np.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    st.image(visualization, caption="AI Attention Heatmap", use_container_width=True)