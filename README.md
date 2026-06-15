# Emotion Recognition from Speech using Deep Learning

## Project Overview

Emotion Recognition from Speech (SER) is a Deep Learning project that automatically identifies human emotions from speech recordings. The system analyzes audio signals, extracts Mel-Frequency Cepstral Coefficients (MFCCs), and classifies emotions using a hybrid CNN-LSTM neural network.

This project was developed as part of the CodeAlpha Machine Learning Internship Program.

---

# Objective

The objective of this project is to detect human emotions from speech audio recordings using Deep Learning techniques.

Recognized emotions include:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

The model learns speech patterns and acoustic features associated with different emotional states.

---

# Dataset

Dataset Used: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Dataset Characteristics:

* 24 Professional Actors
* 8 Emotion Categories
* High Quality WAV Audio Files
* Widely Used Research Dataset

Emotion Labels:

| Code | Emotion   |
| ---- | --------- |
| 01   | Neutral   |
| 02   | Calm      |
| 03   | Happy     |
| 04   | Sad       |
| 05   | Angry     |
| 06   | Fearful   |
| 07   | Disgust   |
| 08   | Surprised |

---

# Technologies Used

* Python
* TensorFlow / Keras
* Librosa
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib

---

# Machine Learning Workflow

## Step 1: Audio Collection

Speech recordings are collected from the RAVDESS dataset.

## Step 2: Audio Processing

The audio files are loaded using Librosa.

Operations performed:

* Audio Loading
* Sampling Rate Handling
* Noise Reduction
* Signal Processing

## Step 3: Feature Extraction

Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from each audio file.

MFCCs help represent speech characteristics in a form suitable for machine learning.

Extracted Features:

* 40 MFCC Features
* Mean Feature Representation
* Standardized Feature Values

## Step 4: Data Preprocessing

* Label Encoding
* Feature Scaling
* Train-Test Splitting
* Reshaping for Deep Learning Models

## Step 5: Model Development

A CNN-LSTM Hybrid Network is used.

### CNN Layer

Extracts local speech patterns and important acoustic features.

### LSTM Layer

Captures temporal dependencies and sequence information in speech.

### Dense Layers

Perform final emotion classification.

---

# Deep Learning Architecture

Input Audio

↓

MFCC Feature Extraction

↓

Feature Scaling

↓

CNN Layer

↓

Max Pooling

↓

CNN Layer

↓

Max Pooling

↓

LSTM Layer

↓

Dropout Layer

↓

Dense Layer

↓

Softmax Output Layer

↓

Emotion Prediction

---

# Model Training

Training Parameters:

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Epochs: 50
* Batch Size: 32
* Early Stopping Enabled

---

# Performance Evaluation

The model is evaluated using:

* Accuracy Score
* Classification Report
* Precision
* Recall
* F1 Score
* Confusion Matrix

Visualization Includes:

* Training Accuracy Curve
* Validation Accuracy Curve
* Confusion Matrix Heatmap

---

# Project Structure

```text
CodeAlpha_Emotion_Recognition/
│
├── data/
│   └── RAVDESS/
│       ├── Actor_01
│       ├── Actor_02
│       ├── Actor_03
│       └── ...
│
├── models/
│   ├── emotion_model.h5
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── emotion_recognition.py
│
├── requirements.txt
│
└── README.md
```

---

# Installation

Clone Repository

```bash
git clone https://github.com/yourusername/CodeAlpha_Emotion_Recognition.git
```

Move to Project Folder

```bash
cd CodeAlpha_Emotion_Recognition
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Execute:

```bash
python emotion_recognition.py
```

The program will:

* Load Audio Files
* Extract MFCC Features
* Train CNN-LSTM Model
* Evaluate Performance
* Generate Visualizations
* Save Trained Model

---

# Output Files

After successful execution:

```text
models/
├── emotion_model.h5
├── scaler.pkl
└── label_encoder.pkl
```

---

# Expected Results

Typical Accuracy:

* CNN Only: 70–75%
* LSTM Only: 72–78%
* CNN + LSTM: 75–85%

Actual performance may vary depending on training configuration and dataset distribution.

---

# Future Enhancements

* Real-Time Voice Emotion Detection
* Streamlit Web Application
* Flask Deployment
* Audio Data Augmentation
* Transformer-Based Models
* Speech-to-Text Integration
* Live Microphone Emotion Recognition

---

# Learning Outcomes

This project demonstrates:

* Audio Signal Processing
* Feature Engineering
* MFCC Extraction
* Deep Learning
* CNN Networks
* LSTM Networks
* Emotion Classification
* Model Evaluation
* Model Deployment Preparation

---

# Conclusion

This project successfully recognizes human emotions from speech using Deep Learning techniques. By combining MFCC-based audio feature extraction with a CNN-LSTM architecture, the system effectively learns emotional patterns from speech signals and provides accurate emotion classification suitable for real-world applications.4




Author

Priyal Parmar 
