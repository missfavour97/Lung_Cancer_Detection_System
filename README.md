# Lung Cancer Detection System

## Overview

This is a school project that demonstrates how deep learning can be used to classify lung CT scan images as `cancer` or `no_cancer`. The project combines image classification, simple lung segmentation, Grad-CAM explainability, PDF report generation, and a small rule-based chatbot inside a Streamlit web app.

The system is built for educational demonstration only. It is not a real medical diagnosis tool.

## Project Aim

The aim of this project is to show an end-to-end AI workflow for medical image analysis:

- Prepare a CT image dataset for training, validation, and testing
- Train a ResNet-18 image classification model
- Display the prediction and confidence score in a web interface
- Use Grad-CAM to show which image regions influenced the model
- Segment the lung area using OpenCV
- Generate a downloadable PDF summary report
- Provide basic patient-support explanations through a chatbot

## Features

- Lung CT image classification using ResNet-18
- OpenCV-based lung region segmentation
- Grad-CAM attention heatmap for model explainability
- Illustrative suspicious-region overlay for presentation purposes
- Hospital-style educational PDF report generation using ReportLab
- System-focused chatbot with similarity-based question matching
- Training loss and validation accuracy graphs
- Data leakage checking scripts for duplicate images

## Technologies Used

- Python
- Streamlit
- PyTorch and Torchvision
- OpenCV
- NumPy
- Pillow
- Matplotlib
- scikit-learn
- Grad-CAM
- ReportLab

## Project Structure

```text
Lung_Cancer_Detection_System/
├── app.py
├── README.md
├── requirements.txt
├── data_leakage.py
├── fix_leakage.py
├── training_loss.png
├── validation_accuracy.png
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   └── best_lung_cancer_model.pth
├── src/
│   ├── segmentation.py
│   ├── split_dataset.py
│   └── train_model.py
└── images/
    └── hero images used by the Streamlit app
```

## Installation

```bash
git clone https://github.com/missfavour97/Lung_Cancer_Detection_System
cd Lung_Cancer_Detection_System
/bin/python -m pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Train the Model

```bash
python3 src/train_model.py
```

The training script uses:

- `dataset/train` for training
- `dataset/val` for validation
- `dataset/test` for final testing

The best model is saved as:

```text
models/best_lung_cancer_model.pth
```

## Model Performance

Reported test performance from the current project run:

```text
Test Accuracy: 97.01%
F1-score: 0.97
Cancer Recall: 1.00

Confusion Matrix:
[[31  0]
 [ 2 34]]
```

These results are for the school project dataset and should not be interpreted as clinical performance.

## App Outputs

After an image is uploaded, the app can show:

- Uploaded CT scan
- Prediction result
- Confidence score
- Confidence category
- Illustrative suspicious-region overlay
- Segmented lung region
- Grad-CAM heatmap
- Downloadable PDF report
- System-focused chatbot explanation

## PDF Report

The app generates a polished educational PDF report with a hospital-style layout. The report includes:

- Original AI Diagnostic Center logo drawn inside the PDF
- Report ID and generation date
- Optional patient name or ID
- Optional age
- Scan date
- Optional doctor or class supervisor name
- Study information section
- Prediction summary
- Confidence score and confidence category
- Original uploaded image
- Segmented lung image
- Grad-CAM heatmap
- Recommendation section
- Educational disclaimer

The logo is custom for this school project and does not represent a real hospital or medical institution.

## System-Focused Chatbot

The chatbot is designed to answer questions about this specific project instead of acting like a general medical chatbot. It can explain:

- The current prediction result
- Confidence score and confidence category
- Grad-CAM heatmap
- Lung segmentation output
- Illustrative suspicious-region overlay
- PDF report
- Dataset structure
- ResNet-18 model
- Accuracy metrics
- Project limitations
- How to run or train the project

The chatbot uses TF-IDF similarity matching, so it can understand different ways of asking the same project-related question. It still gives only educational responses and does not provide real medical diagnosis.

## Hero Slide Images

The files in the `images/` folder are used only for the homepage image slider in the Streamlit app. They make the school project presentation look more complete, but they are not used for training, validation, testing, or prediction.

The app expects images named:

```text
images/hero1.jpg
images/hero2.jpg
images/hero3.jpg
images/hero4.jpg
images/hero5.jpg
images/hero6.jpg
```

For a faster app, these images can be resized or compressed before deployment. If external images are used, their sources should be credited in the project report or presentation.

## Important Note About The Suspicious-Region Overlay

The green box shown in the app is an illustrative suspicious region for presentation purposes. It is not produced by a trained YOLO detector. The Grad-CAM heatmap is the main visual explanation of the classifier's attention.

## Dataset Notes

The dataset is arranged into `cancer` and `no_cancer` folders under train, validation, and test splits. The project also includes scripts to check and remove exact duplicate files between dataset splits:

```bash
python3 data_leakage.py
python3 fix_leakage.py
```

## Limitations

- This is a school project and not a certified medical system.
- The dataset is limited and may not represent real-world clinical variation.
- The app accepts image files such as JPG and PNG, not full clinical DICOM studies.
- The confidence category is based only on model confidence, not clinical cancer staging.
- The suspicious-region box is illustrative, not a real object-detection output.
- A doctor or radiologist must always make real medical decisions.

## Future Improvements

- Use a larger and more diverse dataset
- Add stronger image preprocessing and normalization
- Add data augmentation during training
- Store model class labels separately from the dataset folder
- Add automated tests for model loading, segmentation, and PDF generation
- Add an optional API-based chatbot mode for more natural responses
- Add screenshots of the Streamlit app to the README

## Author

Favour Okwudili  
Computer Engineering Student
