import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random

# --- Config ---
MODEL_PATH = "optical_cnn_model.h5"
OPTICAL_DATA_DIR = "DATA_Maguire_20180517_ALL/SDNET2018/D"  # Adjust if needed
IMG_SIZE = (128, 128)

# --- Load and preprocess images ---
def load_images(label_folder, label_value, limit=30):
    folder_path = os.path.join(OPTICAL_DATA_DIR, label_folder)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected_files = random.sample(image_files, min(limit, len(image_files)))

    images = []
    labels = []
    for fname in selected_files:
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(label_value)
    return np.array(images), np.array(labels)

# --- Main evaluation ---
def evaluate_model():
    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Loading images...")
    X_cd, y_cd = load_images("CD", 1)  # Cracked
    X_ud, y_ud = load_images("UD", 0)  # Uncracked

    X = np.concatenate([X_cd, X_ud], axis=0)
    y = np.concatenate([y_cd, y_ud], axis=0)

    # Add batch dimension
    X = np.expand_dims(X, axis=0) if len(X.shape) == 3 else X

    print("Running predictions...")
    y_pred_prob = model.predict(X)
    # For softmax output (e.g., [0.1, 0.9]) → pick class 1
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\n--- Evaluation Report ---")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print("Classification Report:\n", classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate_model()
