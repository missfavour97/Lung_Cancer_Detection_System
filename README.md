# Lung Cancer Detection System

##  Overview
This project presents a Lung Cancer Detection System that uses deep learning to analyze CT scan images and predict whether cancer is present. The system integrates classification, segmentation, explainability, and a simple chatbot to support user interaction.

##  Features
- Lung cancer classification using ResNet-18  
- Lung segmentation using OpenCV  
- Model explainability using Grad-CAM  
- Performance evaluation with accuracy, precision, recall, F1-score, and confusion matrix  
- Training visualization with loss and accuracy graphs  
- Patient Support AI chatbot for guidance and explanation  

##  Technologies Used
Python, PyTorch, Torchvision, OpenCV, NumPy, Streamlit, Matplotlib, scikit-learn, pytorch-grad-cam  

##  Project Structure
Lung_Cancer_Detection_System/
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   └── best_lung_cancer_model.pth
├── src/
│   └── train_model.py
├── app.py
├── requirements.txt
├── training_loss.png
├── validation_accuracy.png
└── README.md

##  Installation
git clone <https://github.com/missfavour97/Lung_Cancer_Detection_System>  
cd Lung_Cancer_Detection_System  
pip install -r requirements.txt  

##  Run the Application
python3 -m streamlit run app.py  

Open in browser: http://localhost:8501  

##  Train the Model
python3 src/train_model.py  

##  Model Performance
Test Accuracy: 97.01%  
F1-score: 0.97  
Cancer Recall: 1.00  

Confusion Matrix:
[[31  0]
 [ 2 34]]

##  Chatbot Functionality
The chatbot explains predictions, provides guidance, and answers questions about results, accuracy, segmentation, heatmap, and next steps.

##  Disclaimer
This system is for educational purposes only and should not be used for medical diagnosis. Always consult a qualified healthcare professional.

##  Future Improvements
- Use larger and more diverse datasets  
- Improve segmentation with deep learning methods  
- Enhance chatbot with advanced AI models  

##  Author
Favour Okwudili Computer engineering
