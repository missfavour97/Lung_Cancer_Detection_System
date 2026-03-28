import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

st.title("Lung Cancer Detection System")
st.write("Upload an image to test")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("models/best_lung_cancer_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

classes = ["cancer", "no_cancer"]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    prediction = classes[predicted.item()]
    confidence_score = confidence.item() * 100

    if prediction == "no_cancer":
        display_prediction = "No Cancer"
        stage = "No Cancer Detected"
    else:
        display_prediction = "Cancer"
        stage = "Early Stage"

    st.write(f"### Prediction: {display_prediction}")
    st.write(f"### Confidence Score: {confidence_score:.2f}%")
    st.write(f"### Estimated Stage Group: {stage}")
    st.write("Medical Disclaimer: This system is for research purposes only and not for clinical diagnosis.")