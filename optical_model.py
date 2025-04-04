# optical_model.py — Real Optical Classifier (Trained on SDNET2018)

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from image_analysis_model import load_sdnet2018_images, prepare_dataset, get_label_map

MODEL_PATH = "optical_cnn_model.h5"
DATA_PATH = "/Users/gzia/Documents/GitHub/Real-Time Monitoring/DATA_Maguire_20180517_ALL/SDNET2018"
IMG_SIZE = (128, 128)

# ---- MODEL ARCHITECTURE ----
def build_optical_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---- TRAINING ENTRY POINT ----
def train_optical_classifier():
    label_map = get_label_map("optical")
    X, y = load_sdnet2018_images(DATA_PATH, label_map, img_size=IMG_SIZE)

    # ✅ Limit dataset size (for memory-safe training)
    max_samples = 1500
    crack_indices = np.where(y == 1)[0][:max_samples]
    no_crack_indices = np.where(y == 0)[0][:max_samples]
    selected_indices = np.concatenate([crack_indices, no_crack_indices])

    X = X[selected_indices]
    y = y[selected_indices]

    print(f"Training on {len(X)} samples (balanced subset)")

    y_cat = to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    model = build_optical_cnn(input_shape=X_train.shape[1:])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=1)
    model.save(MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")

# ---- PREDICTION API ----
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None
    print("[WARNING] Model file not found. Please train the model using: python optical_model.py")

def predict_crack_from_image(img_array):
    if model is None:
        raise RuntimeError("Model not loaded. Make sure the model file exists.")
    img = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img)
    class_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return class_idx, confidence

# ---- COMMAND LINE ENTRY ----
if __name__ == "__main__":
    train_optical_classifier()