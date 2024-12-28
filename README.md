# Alzheimer-Diagnosis-MRI-Scans-Python-MachineLearning

## 🧠 **Project Overview**
This project utilizes **Machine Learning** and **Deep Learning** techniques to detect **Alzheimer's Disease** using **MRI Scans**. By combining features from **OpenCV** and a pre-trained **VGG16 CNN model**, the system classifies MRI scans as either **Healthy** or **Alzheimer's**. An ensemble learning approach integrates **Random Forest** and **XGBoost** classifiers to enhance prediction accuracy.

The system also provides results in a **visual and intuitive format**, helping to diagnose conditions effectively.

---

## 🔍 **Features**

### **Preprocessing**
- **OpenCV-based Feature Extraction**:
  - **Local Binary Patterns (LBP)**: Captures texture information.
  - **Canny Edge Detection**: Identifies edges in MRI scans.
- **CNN Feature Extraction**:
  - Leverages the **VGG16 model** pre-trained on ImageNet.
- **Image Resizing and Normalization**:
  - Ensures compatibility with the CNN model.

### **Classification**
- Combines features from OpenCV and CNN for robust predictions.
- Implements **Random Forest** and **XGBoost** for classification.
- Employs an **Ensemble Model** with **majority voting** for final predictions.

### **Testing**
- Predicts the condition of **individual MRI images**.
- Handles **batch predictions** for unseen datasets.
- Displays results in a **2x2 grid visualization** with annotated images.

### **Visualization**
- Annotates each image with **predicted results** and file names.
- Provides a **confusion matrix** to summarize classification performance.

---

## 🛠️ **Technologies Used**
- **Python**: Core programming language.
- **OpenCV**: For image preprocessing and feature extraction.
- **Scikit-learn**: For Random Forest implementation.
- **XGBoost**: For gradient boosting classification.
- **TensorFlow/Keras**: For VGG16 feature extraction.
- **Matplotlib** and **Seaborn**: For visualization and plotting.

---
## 📖 How to Run
1. Train the Models
-Execute the Jupyter Notebook or Python script.
-Preprocess the dataset and train:
-Random Forest Classifier
-XGBoost Classifier
-Save trained models for future predictions.
2. Test with Individual Images
-Use the single-image testing functionality to classify a single MRI scan.
3. Test with a Batch of Images
-Load a dataset of unseen MRI images.
-Display the results in a 2x2 grid visualization annotated with predictions and filenames.


## 📊 Project Outputs
Predictions
For unseen MRI images, the predictions are displayed as:
-Healthy
-Alzheimer
Each prediction is annotated alongside the corresponding MRI scan.
Classification Reports
Precision, recall, F1-score, and accuracy metrics are provided for:
Random Forest
XGBoost
Ensemble Model
Confusion Matrix
A detailed confusion matrix is displayed to highlight the performance of the Ensemble Model.
