import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

IMG_SIZE = (128, 128)
IMAGE_DIR = "/Users/gzia/Documents/GitHub/Real-Time Monitoring/Crack_detection/01-Visible images"
MASK_DIR = "/Users/gzia/Documents/GitHub/Real-Time Monitoring/Crack_detection/04-Ground truth"
MODEL_SAVE_PATH = "unet_crack_segmentation.h5"

def load_image_mask_pairs():
    X, y = [], []
    for filename in os.listdir(IMAGE_DIR):
        if not filename.endswith(".png"):
            continue
        base = os.path.splitext(filename)[0]

        image_path = os.path.join(IMAGE_DIR, base + ".png")
        mask_path = os.path.join(MASK_DIR, base + ".jpg")

        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            continue

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        mask = cv2.resize(mask, IMG_SIZE)
        mask = (mask > 127).astype(np.float32)

        X.append(img / 255.0)
        y.append(np.expand_dims(mask, axis=-1))

    return np.array(X), np.array(y)


def build_unet(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)

    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)

    u4 = Conv2DTranspose(32, 2, strides=2, padding='same')(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(u4)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)

    u5 = Conv2DTranspose(16, 2, strides=2, padding='same')(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(16, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_segmentation_model():
    print("Loading data...")
    X, y = load_image_mask_pairs()
    print(f"Loaded {X.shape[0]} image-mask pairs")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_unet(input_shape=X_train.shape[1:])
    model.summary()

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16, callbacks=[checkpoint])

    # Optional: plot training curve
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title("Segmentation Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    train_segmentation_model()
