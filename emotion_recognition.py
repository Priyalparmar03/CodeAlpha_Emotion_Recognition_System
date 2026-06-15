import os
import joblib
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# Emotion Mapping
# ==========================================

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ==========================================
# Feature Extraction
# ==========================================

def extract_features(file_path):

    try:
        audio, sample_rate = librosa.load(
            file_path,
            duration=3,
            offset=0.5
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc_scaled = np.mean(
            mfcc.T,
            axis=0
        )

        return mfcc_scaled

    except Exception as e:
        print("Error:", file_path)
        print(e)
        return None


# ==========================================
# Load Dataset
# ==========================================

features = []
labels = []

dataset_path = "data/RAVDESS"

print("Loading Audio Files...")

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".wav"):

            try:

                emotion_code = file.split("-")[2]

                emotion = emotion_map[
                    emotion_code
                ]

                file_path = os.path.join(
                    root,
                    file
                )

                feature = extract_features(
                    file_path
                )

                if feature is not None:

                    features.append(feature)
                    labels.append(emotion)

            except Exception as e:
                print("Skipping:", file)
                print(e)

print("\nTotal Samples:", len(features))

# ==========================================
# Convert to Arrays
# ==========================================

X = np.array(features)
y = np.array(labels)

print("Feature Shape:", X.shape)

# ==========================================
# Encode Labels
# ==========================================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# ==========================================
# Scale Features
# ==========================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# Reshape for CNN-LSTM
# ==========================================

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

print("Training Shape:", X_train.shape)

# ==========================================
# Build CNN + LSTM Model
# ==========================================

model = Sequential()

model.add(
    Conv1D(
        filters=128,
        kernel_size=3,
        activation='relu',
        input_shape=(
            X_train.shape[1],
            1
        )
    )
)

model.add(
    MaxPooling1D(
        pool_size=2
    )
)

model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu'
    )
)

model.add(
    MaxPooling1D(
        pool_size=2
    )
)

model.add(
    LSTM(
        128,
        return_sequences=False
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        64,
        activation='relu'
    )
)

model.add(
    Dense(
        len(np.unique(y)),
        activation='softmax'
    )
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# Early Stopping
# ==========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ==========================================
# Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_test,
        y_test
    ),
    epochs=50,
    batch_size=32,
    callbacks=[
        early_stop
    ]
)

# ==========================================
# Evaluate
# ==========================================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(
    f"\nTest Accuracy: {accuracy*100:.2f}%"
)

# ==========================================
# Predictions
# ==========================================

predictions = model.predict(
    X_test
)

predictions = np.argmax(
    predictions,
    axis=1
)

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )
)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.title(
    "Emotion Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()

# ==========================================
# Training Accuracy Graph
# ==========================================

plt.figure(
    figsize=(8,5)
)

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# ==========================================
# Save Model
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/emotion_model.h5"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

print("\nModel Saved Successfully!")
print("emotion_model.h5")
print("scaler.pkl")
print("label_encoder.pkl")