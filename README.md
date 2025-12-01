Plant Disease Classification Using Deep Learning
------------------------------------------------

This repository contains the complete source code for my CSC 590 Master's Project:
"Plant Disease Classification Using Deep Learning"
by Sonali Rangampally, 
California State University, Dominguez Hills.

The project uses deep learning (Convolutional Neural Networks) to classify plant leaf diseases
across 38 classes using the "New Plant Diseases Dataset" from Kaggle.

------------------------------------------------
1. Dataset
------------------------------------------------
- Dataset: New Plant Diseases Dataset (Kaggle)
- Total images: ~87,000
- Classes: 38 disease categories
- Plants include: Tomato, Apple, Potato, Grape, Corn, etc.
- The dataset was split into training, validation, and test sets using stratified splitting.

------------------------------------------------
2. Project Pipeline
------------------------------------------------
The workflow for this project is:

Data Collection
→ Image Preprocessing
→ Data Augmentation
→ Model Selection
→ Training & Fine-tuning
→ Evaluation
→ Explainability (Grad-CAM)
→ Deployment using TensorFlow Lite

------------------------------------------------
3. Models Used
------------------------------------------------
The following pre-trained architectures were trained and compared:
- VGG16
- ResNet50
- DenseNet121
- MobileNetV2
- EfficientNet-B3

All models were trained using transfer learning and fine-tuned for the dataset.

------------------------------------------------
4. How to Run the Project
------------------------------------------------
1. Install required libraries:
   pip install -r requirements.txt

2. Open the notebook:
   Plant_disease_classification.ipynb

3. Run all cells in order to:
   - Preprocess data
   - Train the models
   - Evaluate accuracy, precision, recall, F1-score
   - Generate confusion matrix
   - Produce Grad-CAM explainability heatmaps
   - Export TensorFlow Lite model

------------------------------------------------
5. Results Summary
------------------------------------------------
Best performing model: EfficientNet-B3

Validation Accuracy: 96.3%
Precision: 0.96
Recall: 0.96
F1 Score: 0.96

DenseNet121 also performed strongly.
ResNet50 and VGG16 performed lower compared to EfficientNet.

------------------------------------------------
6. Explainability
------------------------------------------------
Grad-CAM visualizations were used to understand model focus.
Correct predictions show heatmaps on lesion regions.
Incorrect predictions sometimes show attention on background regions.

------------------------------------------------
7. Deployment
------------------------------------------------
The best model (EfficientNet-B3) was converted to TensorFlow Lite (.tflite)
for running on mobile or edge devices.

------------------------------------------------
8. Folder Structure
------------------------------------------------

Plant-Disease-Classification-Using-Deep-Learning/
│
├── Plant_disease_classification.ipynb
├── requirements.txt
├── README.txt
│

------------------------------------------------
9. Acknowledgements
------------------------------------------------
- CSUDH Computer Science Department
- Project Advisor / Committee
- Authors of CNN architectures used
- Kaggle dataset contributors

------------------------------------------------
End of README.txt
------------------------------------------------
