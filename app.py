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
import base64
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import tempfile
import io
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(layout="wide")


@st.cache_data(show_spinner=False)
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_slider_keyframes(image_count):
    if image_count <= 1:
        return "0% { transform: translateX(0%); } 100% { transform: translateX(0%); }"

    keyframes = []
    step = 100 / image_count
    hold = step * 0.82

    for index in range(image_count):
        position = -(index * step)
        start = index * step
        end = min(start + hold, 100)

        keyframes.append(f"{start:.2f}% {{ transform: translateX({position:.2f}%); }}")
        keyframes.append(f"{end:.2f}% {{ transform: translateX({position:.2f}%); }}")

    keyframes.append("100% { transform: translateX(0%); }")
    return "\n".join(keyframes)


def is_likely_ct_scan(image):
    img_np = np.array(image)

    # Check if image is mostly grayscale
    if len(img_np.shape) == 3:
        r = img_np[:, :, 0]
        g = img_np[:, :, 1]
        b = img_np[:, :, 2]

        color_difference = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b))

        if color_difference > 25:
            return False

    # Check brightness/contrast range
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    contrast = gray.std()

    if contrast < 10:
        return False

    return True


def image_to_pdf_reader(image):
    if isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.fromarray(np.asarray(image))

    if pil_image.mode not in ("RGB", "RGBA"):
        pil_image = pil_image.convert("RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return ImageReader(buffer)


def draw_report_logo(c, x, y, size=54):
    center_x = x + size / 2
    center_y = y + size / 2

    c.setFillColorRGB(1, 1, 1)
    c.circle(center_x, center_y, size / 2, fill=1, stroke=0)

    c.setStrokeColorRGB(0.0, 0.38, 0.62)
    c.setLineWidth(2)
    c.line(center_x, center_y + 14, center_x, center_y + 2)
    c.line(center_x, center_y + 2, center_x - 7, center_y - 6)
    c.line(center_x, center_y + 2, center_x + 7, center_y - 6)

    c.setFillColorRGB(0.82, 0.95, 0.98)
    c.setStrokeColorRGB(0.0, 0.38, 0.62)
    c.ellipse(center_x - 18, center_y - 16, center_x - 2, center_y + 9, fill=1, stroke=1)
    c.ellipse(center_x + 2, center_y - 16, center_x + 18, center_y + 9, fill=1, stroke=1)

    c.setFillColorRGB(0.0, 0.55, 0.68)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(center_x, y + 6, "AI")


def draw_section_title(c, title, x, y):
    c.setFillColorRGB(0.0, 0.22, 0.36)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, title)
    c.setStrokeColorRGB(0.78, 0.86, 0.90)
    c.setLineWidth(0.8)
    c.line(x, y - 6, x + 515, y - 6)


def draw_info_field(c, label, value, x, y):
    c.setFillColorRGB(0.42, 0.48, 0.52)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x, y, label.upper())
    c.setFillColorRGB(0.08, 0.12, 0.16)
    c.setFont("Helvetica", 10)
    c.drawString(x, y - 15, value)


def draw_wrapped_text(c, text, x, y, max_width, line_height=12, font_name="Helvetica", font_size=9):
    c.setFont(font_name, font_size)
    words = text.split()
    line = ""

    for word in words:
        test_line = f"{line} {word}".strip()
        if c.stringWidth(test_line, font_name, font_size) <= max_width:
            line = test_line
        else:
            c.drawString(x, y, line)
            y -= line_height
            line = word

    if line:
        c.drawString(x, y, line)

    return y - line_height


def draw_image_panel(c, title, image_reader, x, y, width, height):
    c.setFillColorRGB(0.97, 0.99, 1.0)
    c.setStrokeColorRGB(0.80, 0.88, 0.92)
    c.roundRect(x, y, width, height, 8, fill=1, stroke=1)

    c.setFillColorRGB(0.0, 0.22, 0.36)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 10, y + height - 20, title)

    c.drawImage(
        image_reader,
        x + 10,
        y + 12,
        width=width - 20,
        height=height - 42,
        preserveAspectRatio=True,
        anchor="c",
    )


def clean_report_field(value, fallback="Not provided"):
    value = str(value).strip()
    return value if value else fallback


def create_pdf_report(
    prediction,
    confidence_score,
    confidence_category,
    original_img,
    segmentation_img,
    heatmap_img,
    report_details=None,
):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = temp_file.name
    generated_at = datetime.now()
    report_id = generated_at.strftime("ADC-%Y%m%d-%H%M%S")
    report_details = report_details or {}
    patient_name = clean_report_field(report_details.get("patient_name"))
    patient_age = clean_report_field(report_details.get("patient_age"))
    scan_date = clean_report_field(report_details.get("scan_date"))
    supervisor_name = clean_report_field(report_details.get("supervisor_name"))

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 40
    content_width = width - (margin * 2)

    original_reader = image_to_pdf_reader(original_img)
    seg_reader = image_to_pdf_reader(segmentation_img)
    heat_reader = image_to_pdf_reader(heatmap_img)

    # Hospital-style header
    c.setFillColorRGB(0.0, 0.22, 0.36)
    c.rect(0, height - 105, width, 105, fill=1, stroke=0)
    c.setFillColorRGB(0.0, 0.58, 0.70)
    c.rect(0, height - 109, width, 4, fill=1, stroke=0)

    draw_report_logo(c, margin, height - 86)

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(110, height - 43, "AI Diagnostic Center")

    c.setFont("Helvetica", 12)
    c.drawString(112, height - 64, "Educational Lung CT Screening Report")

    c.setFont("Helvetica-Bold", 9)
    c.drawRightString(width - margin, height - 42, "REPORT ID")
    c.setFont("Helvetica", 9)
    c.drawRightString(width - margin, height - 56, report_id)
    c.drawRightString(width - margin, height - 72, generated_at.strftime("%Y-%m-%d %H:%M:%S"))

    c.setFillColorRGB(0.88, 0.97, 1.0)
    c.roundRect(width - margin - 120, height - 96, 120, 18, 8, fill=1, stroke=0)
    c.setFillColorRGB(0.0, 0.30, 0.46)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(width - margin - 60, height - 91, "EDUCATIONAL PROTOTYPE")

    # Study information
    draw_section_title(c, "Study Information", margin, height - 132)
    c.setFillColorRGB(1, 1, 1)
    c.setStrokeColorRGB(0.82, 0.88, 0.91)
    c.roundRect(margin, height - 218, content_width, 64, 8, fill=1, stroke=1)

    draw_info_field(c, "Patient Name / ID", patient_name, margin + 18, height - 176)
    draw_info_field(c, "Age", patient_age, margin + 170, height - 176)
    draw_info_field(c, "Scan Date", scan_date, margin + 245, height - 176)
    draw_info_field(c, "Doctor / Supervisor", supervisor_name, margin + 365, height - 176)

    # Prediction summary
    draw_section_title(c, "Prediction Summary", margin, height - 246)
    c.setFillColorRGB(0.97, 0.99, 1.0)
    c.setStrokeColorRGB(0.82, 0.88, 0.91)
    c.roundRect(margin, height - 342, content_width, 76, 8, fill=1, stroke=1)

    result_color = (0.72, 0.12, 0.12) if prediction == "cancer" else (0.08, 0.48, 0.22)
    c.setFillColorRGB(*result_color)
    c.roundRect(margin + 18, height - 315, 130, 28, 12, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(margin + 83, height - 306, prediction.upper())

    draw_info_field(c, "Confidence Score", f"{confidence_score:.2f}%", margin + 180, height - 293)
    draw_info_field(c, "Confidence Category", confidence_category, margin + 315, height - 293)

    c.setFillColorRGB(0.30, 0.35, 0.38)
    summary_note = (
        "This result is produced by a ResNet-18 classifier for cancer / no_cancer image classification and should "
        "be used only for educational demonstration."
    )
    draw_wrapped_text(c, summary_note, margin + 18, height - 330, content_width - 36)

    # Visual findings
    draw_section_title(c, "Visual Findings", margin, height - 372)
    panel_width = 160
    panel_height = 148
    panel_y = height - 544
    draw_image_panel(c, "Original Upload", original_reader, margin, panel_y, panel_width, panel_height)
    draw_image_panel(c, "Segmented Lung Region", seg_reader, margin + 178, panel_y, panel_width, panel_height)
    draw_image_panel(c, "Grad-CAM Heatmap", heat_reader, margin + 356, panel_y, panel_width, panel_height)

    # Recommendation
    draw_section_title(c, "Recommendation", margin, height - 575)
    c.setFillColorRGB(1, 1, 1)
    c.setStrokeColorRGB(0.82, 0.88, 0.91)
    c.roundRect(margin, height - 648, content_width, 50, 8, fill=1, stroke=1)

    if prediction == "cancer":
        recommendation = (
            "The model detected cancer-like patterns. For a real case, the scan would need review by a qualified "
            "doctor or radiologist. For this project, review the heatmap, segmentation, and confidence score."
        )
    else:
        recommendation = (
            "No strong cancer-like pattern was detected by the model. Review the heatmap, "
            "segmentation, and confidence score as supporting outputs."
        )

    c.setFillColorRGB(0.15, 0.18, 0.20)
    draw_wrapped_text(c, recommendation, margin + 18, height - 620, content_width - 36, line_height=11)

    # Disclaimer
    draw_section_title(c, "Educational Disclaimer", margin, height - 676)
    c.setFillColorRGB(1.0, 0.97, 0.88)
    c.setStrokeColorRGB(0.90, 0.76, 0.35)
    c.roundRect(margin, height - 744, content_width, 48, 8, fill=1, stroke=1)

    disclaimer = (
        "This AI-generated report is part of a school project. It is not a certified medical report, does not confirm "
        "diagnosis, and must not be used for real medical decisions."
    )
    c.setFillColorRGB(0.30, 0.22, 0.06)
    draw_wrapped_text(c, disclaimer, margin + 18, height - 716, content_width - 36, line_height=11)

    # Footer
    c.setStrokeColorRGB(0.82, 0.88, 0.91)
    c.line(margin, 54, width - margin, 54)

    c.setFillColorRGB(0.0, 0.22, 0.36)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin, 36, "AI Diagnostic Center")

    c.setFillColorRGB(0.42, 0.48, 0.52)
    c.setFont("Helvetica", 9)
    c.drawRightString(width - margin, 36, "Generated by Lung Cancer Detection System")

    c.save()
    return pdf_path


@st.cache_resource(show_spinner="Loading classification model...")
def load_classifier():
    dataset = datasets.ImageFolder("dataset/train")
    classes = dataset.classes

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("models/best_lung_cancer_model.pth", map_location="cpu"))
    model.eval()

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    return model, classes, cam


def answer_result(context):
    if not context["has_result"]:
        return "Please upload a CT scan image first. After prediction, I can explain the result in simple terms."

    if context["prediction"] == "cancer":
        return (
            f"The system predicted `cancer` with a confidence score of {context['confidence']:.2f}%. "
            f"The confidence category is `{context['confidence_category']}`. This means the model found image patterns "
            "similar to the cancer examples in the school project dataset. It is not a medical diagnosis."
        )

    return (
        f"The system predicted `no_cancer` with a confidence score of {context['confidence']:.2f}%. "
        "This means the model did not find strong cancer-like patterns in the uploaded image, based on what it learned "
        "from the project dataset."
    )


def answer_confidence(context):
    if not context["has_result"]:
        return "Upload a CT scan first so the system can calculate a confidence score."

    return (
        f"The confidence score is {context['confidence']:.2f}%. It shows how strongly the model supports its selected "
        "class for this image. A high score does not mean a confirmed diagnosis; it only reflects the model output."
    )


def answer_confidence_category(context):
    if not context["has_result"]:
        return "The confidence category appears after an image is uploaded and classified."

    if context["prediction"] == "no_cancer":
        return "The confidence category is not applied because the current prediction is `no_cancer`."

    return (
        f"The confidence category is `{context['confidence_category']}`. It is based only on the model confidence score, "
        "not on clinical lung cancer staging."
    )


def answer_heatmap(context):
    return (
        "The heatmap is generated with Grad-CAM. Warmer areas show the parts of the CT image that influenced the "
        "ResNet-18 classifier most strongly. It helps explain the model's attention, but it should not be treated as "
        "a doctor's marked diagnosis."
    )


def answer_segmentation(context):
    return (
        "The segmentation view uses OpenCV image processing to isolate the lung region. It converts the image to "
        "grayscale, applies blur and thresholding, cleans the mask, and shows the masked lung area."
    )


def answer_overlay(context):
    return (
        "The green box is an illustrative suspicious-region overlay for presentation purposes. It is not produced by "
        "a trained object detector. The Grad-CAM heatmap is the actual model-attention visualization."
    )


def answer_pdf(context):
    return (
        "The PDF report uses a hospital-style layout with an original AI Diagnostic Center logo, report ID, study "
        "details, prediction summary, original image, segmentation image, heatmap, recommendation, and educational "
        "disclaimer. It is meant as a polished school-project output, not a clinical report."
    )


def answer_model(context):
    return (
        "The classification model is ResNet-18 from Torchvision. The final layer is changed to output two classes: "
        "`cancer` and `no_cancer`. The training script saves the best validation model as "
        "`models/best_lung_cancer_model.pth`."
    )


def answer_dataset(context):
    return (
        "The dataset is arranged with separate `train`, `val`, and `test` folders. Each split contains `cancer` and "
        "`no_cancer` classes. The project also includes scripts to check and remove exact duplicate images between "
        "splits."
    )


def answer_accuracy(context):
    return (
        "The README reports 97.01% test accuracy, F1-score of 0.97, and cancer recall of 1.00 for the current school "
        "project dataset. These numbers are useful for the project report, but they should not be interpreted as "
        "real clinical performance."
    )


def answer_next_steps(context):
    if not context["has_result"]:
        return "First upload a CT image. Then I can explain the prediction, heatmap, segmentation, and report output."

    if context["prediction"] == "cancer":
        return (
            "For the project demo, the next step is to review the confidence score, Grad-CAM heatmap, segmentation, and "
            "PDF report. In real life, a suspicious scan must be reviewed by a qualified doctor or radiologist."
        )

    return (
        "For the project demo, you can review the confidence score, heatmap, segmentation, and PDF report. In real life, "
        "a doctor should still review scans if symptoms or concerns continue."
    )


def answer_upload(context):
    return (
        "The app accepts JPG, JPEG, and PNG image uploads. It also checks whether the image looks like a CT scan by "
        "looking for mostly grayscale pixels and enough contrast."
    )


def answer_limitations(context):
    return (
        "This is a school project, so its main limitations are the small dataset, basic image preprocessing, no DICOM "
        "support, no external clinical validation, and an illustrative region box that is not a real detector output."
    )


def answer_run_project(context):
    return (
        "To run the app, install the dependencies with `pip install -r requirements.txt`, then start Streamlit with "
        "`python3 -m streamlit run app.py`. To retrain the model, run `python3 src/train_model.py`."
    )


def answer_symptoms(context):
    return (
        "Common lung cancer symptoms can include persistent cough, chest pain, coughing up blood, shortness of breath, "
        "unexplained weight loss, and fatigue. This project is not a "
        "medical diagnosis system. Always consult a doctor for any health concerns or symptoms you may have."
    )


def answer_risk_factors(context):
    return (
        "Common risk factors include smoking, secondhand smoke, family history, air pollution, and exposure to harmful "
        "substances such as asbestos. The app itself does not calculate personal risk; it only classifies the uploaded image."
    )


def answer_treatment(context):
    return (
        "Treatment decisions are outside this system. Real treatment can depend on clinical evaluation, imaging, biopsy, "
        "and doctor review. In this project, the chatbot only explains the AI output and general concepts."
    )


def answer_capabilities(context):
    return (
        "I can answer questions about this project: the prediction, confidence score, confidence category, heatmap, "
        "segmentation, illustrative region box, PDF report, dataset, model, accuracy, limitations, and how to run it."
    )


CHATBOT_TOPICS = [
    {
        "name": "result",
        "questions": [
            "explain my result prediction outcome what does the result mean why cancer no cancer",
            "why did the system say cancer what does no cancer mean interpret prediction",
        ],
        "answer": answer_result,
    },
    {
        "name": "confidence",
        "questions": [
            "confidence score probability percentage how sure is the model what does confidence mean",
            "is this score reliable how strong is prediction",
        ],
        "answer": answer_confidence,
    },
    {
        "name": "confidence category",
        "questions": [
            "confidence category high model confidence moderate model confidence stage staging",
            "why does it say high confidence category is this cancer stage",
        ],
        "answer": answer_confidence_category,
    },
    {
        "name": "heatmap",
        "questions": [
            "heatmap grad cam attention map red area yellow area why model focus explain visualization",
            "what does ai attention heatmap show",
        ],
        "answer": answer_heatmap,
    },
    {
        "name": "segmentation",
        "questions": [
            "segmentation segmented lung region mask open cv threshold lung area image processing",
            "how does the app isolate lungs what is segmented image",
        ],
        "answer": answer_segmentation,
    },
    {
        "name": "overlay",
        "questions": [
            "green box suspicious region yolo detector bounding box illustrative region detection box",
            "is the box real why is there a rectangle",
        ],
        "answer": answer_overlay,
    },
    {
        "name": "pdf",
        "questions": [
            "pdf report download medical report what is inside report generate report summary",
            "how does the report work",
        ],
        "answer": answer_pdf,
    },
    {
        "name": "model",
        "questions": [
            "model architecture resnet resnet18 pytorch torchvision classifier training model layers",
            "what algorithm does the project use",
        ],
        "answer": answer_model,
    },
    {
        "name": "dataset",
        "questions": [
            "dataset train validation test split cancer no cancer folders data leakage duplicate images",
            "how is data arranged what data did you use",
        ],
        "answer": answer_dataset,
    },
    {
        "name": "accuracy",
        "questions": [
            "accuracy f1 recall precision confusion matrix performance reliable test result metrics",
            "how accurate is the model",
        ],
        "answer": answer_accuracy,
    },
    {
        "name": "next steps",
        "questions": [
            "next step what should i do now after prediction recommendation doctor radiologist review",
            "what happens after upload what do i do with result",
        ],
        "answer": answer_next_steps,
    },
    {
        "name": "upload",
        "questions": [
            "upload image jpg jpeg png ct scan valid invalid image grayscale contrast file type",
            "why was my image rejected what files can i upload",
        ],
        "answer": answer_upload,
    },
    {
        "name": "limitations",
        "questions": [
            "limitations disclaimer school project not real diagnosis clinical validation dicom small dataset",
            "can this be used in hospital real world certified medical system",
        ],
        "answer": answer_limitations,
    },
    {
        "name": "run project",
        "questions": [
            "run app install requirements streamlit train model command start project setup",
            "how do i run this project how do i train it",
        ],
        "answer": answer_run_project,
    },
    {
        "name": "symptoms",
        "questions": [
            "symptoms signs cough chest pain blood shortness breath weight loss fatigue",
            "what are common symptoms",
        ],
        "answer": answer_symptoms,
    },
    {
        "name": "risk factors",
        "questions": [
            "risk factors smoking secondhand smoke family history air pollution asbestos causes",
            "what causes lung cancer risk",
        ],
        "answer": answer_risk_factors,
    },
    {
        "name": "treatment",
        "questions": [
            "treatment therapy surgery chemotherapy radiation biopsy doctor diagnosis medical care",
            "how is lung cancer treated",
        ],
        "answer": answer_treatment,
    },
    {
        "name": "capabilities",
        "questions": [
            "help what can you do chatbot assistant questions explain project system features",
            "what can i ask you",
        ],
        "answer": answer_capabilities,
    },
]


def get_chatbot_response(question, context):
    normalized_question = question.strip().lower()

    if normalized_question in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]:
        return "Hello. I can help explain this lung cancer detection school project and its current prediction output."

    documents = [" ".join(topic["questions"]) for topic in CHATBOT_TOPICS]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    topic_matrix = vectorizer.fit_transform(documents)
    question_vector = vectorizer.transform([normalized_question])
    scores = cosine_similarity(question_vector, topic_matrix).flatten()
    best_index = int(scores.argmax())
    best_score = scores[best_index]

    if best_score < 0.12:
        return (
            "I am focused on this lung cancer detection project. You can ask me about the prediction, confidence, "
            "heatmap, segmentation, PDF report, dataset, model, accuracy, limitations, or how to run the system."
        )

    topic = CHATBOT_TOPICS[best_index]
    return topic["answer"](context)


st.title("Lung Cancer Detection System")

hero_images = [
    "images/hero1.jpg",
    "images/hero2.jpg",
    "images/hero3.jpg",
    "images/hero4.jpg",
    "images/hero5.jpg",
    "images/hero6.jpg",
]

hero_images = [img for img in hero_images if Path(img).exists()]

if hero_images:
    encoded_images = [get_base64_image(img) for img in hero_images]
    slider_keyframes = build_slider_keyframes(len(encoded_images))

    slider_html = f"""
<div style="width:100%; max-width:2000px; margin:auto; overflow:hidden; border-radius:18px; box-shadow:0 6px 18px rgba(0,0,0,0.2);">
  <div class="slider">
    {''.join([f'<img src="data:image/jpg;base64,{img}" class="slide">' for img in encoded_images])}
  </div>
</div>

<style>
.slider {{
  display: flex;
  width: {len(encoded_images) * 100}%;
  height: 360px;
  animation: slideAnimation {len(encoded_images) * 4}s infinite;
  border-radius: 18px;
}}

.slide {{
  width: {100 / len(encoded_images)}%;
  height: 360px;
  object-fit: cover;
  object-position: center center;
  flex-shrink: 0;
  border-radius: 18px;
}}

@keyframes slideAnimation {{
  {slider_keyframes}
}}
</style>
"""

    st.html(slider_html)
else:
    st.info("Hero slide images are missing. Add images named hero1.jpg to hero6.jpg inside the images folder.")

st.markdown("### AI-Based Analysis of Lung CT Scans")
st.write("Upload a CT scan image to receive a class prediction, visual explanations, and patient support guidance.")


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load trained classifier and Grad-CAM once so the chatbot stays responsive during Streamlit reruns
model, classes, cam = load_classifier()

# Upload image
uploaded_file = st.file_uploader("Upload a CT scan image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded CT Scan", width="stretch")

    # Check if the uploaded image is likely a CT scan
    if not is_likely_ct_scan(image):
        st.error("Invalid image. Please upload a valid CT scan image.")
        st.stop()

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
    confidence_score = min(confidence_score, 99.5)

    # Show prediction
    st.write(f"### Prediction: {prediction}")
    st.write(f"### Confidence Score: {confidence_score:.2f}%")

    # Confidence category for the school-project report
    if prediction == "cancer":
        if confidence_score > 85:
            confidence_category = "High model confidence"
        else:
            confidence_category = "Moderate model confidence"

        st.write(f"### Confidence Category: {confidence_category}")
    else:
        confidence_category = "Not applicable"
        st.success("No cancer detected")

    # Medical disclaimer
    st.info("⚠️ This system is for educational purposes only and should not be used for medical diagnosis.")

    # Show an illustrative suspicious region only if cancer is predicted.
    # This is a demo overlay, not a separate object-detection model.
    if prediction == "cancer":
        img_with_boxes = np.array(image).copy()
        h, w, _ = img_with_boxes.shape

        x1 = int(w * 0.30)
        y1 = int(h * 0.25)
        x2 = int(w * 0.70)
        y2 = int(h * 0.65)

        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        st.image(img_with_boxes, caption="Illustrative Suspicious Region (Demo)", width="stretch")
        st.caption("The green box is an illustrative region for presentation purposes. The Grad-CAM heatmap shows the model attention.")
    else:
        st.info("No illustrative suspicious region displayed because the scan was predicted as no cancer.")

    # Show segmentation
    st.image(lung_mask, caption="Segmented Lung Region", width="stretch")

    # Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=img)[0]
    image_np = np.array(image.resize((224, 224))) / 255.0

    visualization = show_cam_on_image(
        image_np.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    st.image(visualization, caption="AI Attention Heatmap", width="stretch")

    # Optional report details for the downloadable PDF
    st.markdown("### Report Details")
    st.caption("Optional fields for the educational PDF report.")

    report_col_1, report_col_2 = st.columns(2)
    patient_name = report_col_1.text_input("Patient name or ID", placeholder="Demo Patient / ID")
    patient_age = report_col_2.text_input("Age", placeholder="Not provided")

    report_col_3, report_col_4 = st.columns(2)
    scan_date = report_col_3.date_input("Scan date", value=datetime.now().date())
    supervisor_name = report_col_4.text_input("Doctor / class supervisor name", placeholder="Not provided")

    report_details = {
        "patient_name": patient_name,
        "patient_age": patient_age,
        "scan_date": scan_date.strftime("%Y-%m-%d"),
        "supervisor_name": supervisor_name,
    }

    # Create PDF Report
    pdf_path = create_pdf_report(
        prediction,
        confidence_score,
        confidence_category,
        image,
        lung_mask,
        visualization,
        report_details,
    )

    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Download Educational PDF Report",
            data=pdf_file,
            file_name="lung_cancer_report.pdf",
            mime="application/pdf"
        )


st.subheader("Patient Support AI Chatbot")

st.caption("Ask focused questions about this system, its prediction, model, heatmap, segmentation, report, dataset, or limitations.")

current_prediction = prediction if "prediction" in locals() else None
current_confidence = confidence_score if "confidence_score" in locals() else 0
current_confidence_category = confidence_category if "confidence_category" in locals() else "Not available"

chatbot_context = {
    "has_result": current_prediction is not None,
    "prediction": current_prediction,
    "confidence": current_confidence,
    "confidence_category": current_confidence_category,
}

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello. I am the project assistant. Ask me about the prediction, confidence score, heatmap, "
                "segmentation, PDF report, dataset, model, accuracy, or limitations."
            ),
        }
    ]

quick_prompt = None
quick_cols = st.columns(4)

if quick_cols[0].button("Explain result"):
    quick_prompt = "Explain my result"
if quick_cols[1].button("Heatmap"):
    quick_prompt = "What does the heatmap show?"
if quick_cols[2].button("Model"):
    quick_prompt = "What model does this system use?"
if quick_cols[3].button("Limitations"):
    quick_prompt = "What are the limitations?"

typed_prompt = st.chat_input("Ask about this lung cancer detection system")
user_prompt = quick_prompt or typed_prompt

if user_prompt:
    st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": get_chatbot_response(user_prompt, chatbot_context),
        }
    )

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
