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

    st.subheader("Patient Support AI Chatbot")

user_input = st.text_input("Ask a question about the result:")

if user_input:
    question = user_input.lower().strip()

    # Make prediction label safe for chatbot logic
    current_prediction = prediction if 'prediction' in locals() else "unknown"
    current_confidence = confidence_score if 'confidence_score' in locals() else 0
    current_stage = stage if 'stage' in locals() else "Not available"

    response = ""

    # Result meaning
    if any(word in question for word in ["what does this mean", "what does the result mean", "explain result", "prediction mean"]):
        if current_prediction == "cancer":
            response = (
                f"The system predicts cancer with a confidence score of {current_confidence:.2f}%. "
                f"The estimated stage is {current_stage}. This is only an AI-based assessment and must be confirmed by a medical professional."
            )
        elif current_prediction == "no_cancer":
            response = (
                f"The system predicts no cancer with a confidence score of {current_confidence:.2f}%. "
                "This means the model did not detect patterns associated with cancer in this scan, but a doctor should still confirm the result."
            )
        else:
            response = "The result is not currently available. Please upload a CT scan first."

    # Cancer questions
    elif "cancer" in question and "no cancer" not in question:
        response = (
            "Cancer refers to abnormal cell growth. In this system, a cancer prediction means the model detected image patterns similar to cancer cases in the training data."
        )

    # Normal / no cancer
    elif "no cancer" in question or "normal" in question:
        if current_prediction == "no_cancer":
            response = (
                "The system predicted no cancer for this scan. That means no strong cancer-related pattern was detected by the model."
            )
        else:
            response = (
                "This scan was not predicted as no cancer. The model detected suspicious patterns, so professional medical review is recommended."
            )

    # Accuracy / trust
    elif any(word in question for word in ["accurate", "accuracy", "trust", "reliable", "can i trust this"]):
        response = (
            "This system performed well on the test dataset, but it is still an educational AI system and not a replacement for medical diagnosis."
        )

    # Confidence
    elif "confidence" in question:
        response = (
            f"The confidence score for this result is {current_confidence:.2f}%. "
            "This shows how strongly the model supports its prediction, but confidence is not the same as a confirmed diagnosis."
        )

    # Stage
    elif "stage" in question:
        if current_prediction == "cancer":
            response = (
                f"The estimated stage shown by the system is {current_stage}. "
                "This is only a simple AI-based indication and not a clinically confirmed cancer stage."
            )
        else:
            response = "No stage is shown because the current prediction is no cancer."

    # Next step
    elif any(word in question for word in ["next", "what should i do", "what next", "what do i do now"]):
        if current_prediction == "cancer":
            response = (
                "The next step is to consult a qualified doctor or radiologist for proper medical evaluation and possible follow-up tests."
            )
        elif current_prediction == "no_cancer":
            response = (
                "Even though the system predicted no cancer, you should still consult a doctor if symptoms persist or if this scan is clinically important."
            )
        else:
            response = "Please upload a CT scan first so the system can generate a result."

    # Heatmap / Grad-CAM
    elif "heatmap" in question or "grad-cam" in question or "why did the model focus" in question:
        response = (
            "The heatmap shows the region the AI focused on most strongly while making its prediction. It helps explain why the model gave that result."
        )

    # Segmentation
    elif "segmentation" in question or "lung region" in question:
        response = (
            "Segmentation isolates the lung region so the system can focus on the most relevant part of the CT scan and reduce background noise."
        )

    # YOLO
    elif "yolo" in question or "bounding box" in question or "box" in question:
        response = (
            "The YOLO-style box in this project is a demonstration of localization. It is used to show how suspicious regions could be highlighted in a more advanced system."
        )

    # Data leakage
    elif "data leakage" in question or "leakage" in question:
        response = (
            "Data leakage happens when similar or duplicate images appear across training and testing sets, causing misleadingly high performance. "
            "This project fixed that issue before final evaluation."
        )

    # Metrics
    elif any(word in question for word in ["f1", "precision", "recall", "confusion matrix", "metrics"]):
        response = (
            "The system was evaluated using accuracy, precision, recall, F1-score, and confusion matrix to measure how well it performs on unseen CT scans."
        )

    # Disclaimer / safety
    elif any(word in question for word in ["disclaimer", "safe", "medical diagnosis", "doctor"]):
        response = (
            "This system is for educational purposes only and should not be used as a final medical diagnosis. A qualified medical professional should always make the final decision."
        )

    # Greeting
    elif any(word in question for word in ["hello", "hi", "hey"]):
        response = "Hello. I can help explain the prediction, confidence score, stage, heatmap, segmentation, YOLO, and next steps."

    # Fallback
    else:
        response = (
            "I can help explain the prediction, confidence score, estimated stage, heatmap, segmentation, YOLO, evaluation metrics, or what to do next."
        )

    st.write("### Chatbot Response")
    st.write(response)