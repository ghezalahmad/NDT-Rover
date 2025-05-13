import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# ---- CONFIG ----
CD_PATH = "DATA_Maguire_20180517_ALL/SDNET2018/D/CD"
UD_PATH = "DATA_Maguire_20180517_ALL/SDNET2018/D/UD"
IMG_SIZE = (128, 128)
MODEL_SAVE_PATH = "optical_binary_classifier.h5"
EPOCHS = 15
BATCH_SIZE = 32

# ---- Load SDNET Binary Dataset ----
def load_sdnet_binary_dataset(cd_path, ud_path, img_size=(128, 128)):
    X, y = [], []
    for folder, label in [(cd_path, 1), (ud_path, 0)]:
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(folder, fname)
                img = cv2.imread(fpath)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    X.append(img / 255.0)
                    y.append(label)
    return np.array(X), np.array(y)

# ---- Build CNN ----
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---- Train and Save ----
if __name__ == "__main__":
    print("Loading data...")
    X, y = load_sdnet_binary_dataset(CD_PATH, UD_PATH, IMG_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model(X_train.shape[1:])
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', verbose=1)
    
    print("Training...")
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[checkpoint],
              verbose=2)
    
    print("Evaluating...")
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {acc:.4f}")
