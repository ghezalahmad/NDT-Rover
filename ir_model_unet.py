# ir_model.py — U-Net Inference for Infrared Crack Segmentation

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load your trained U-Net model
UNET_PATH = "ir_unet_model.h5"
unet_model = load_model(UNET_PATH)

IMG_SIZE = (128, 128)

def preprocess_ir_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    return arr, img

def predict_ir_mask(img_array):
    img = np.expand_dims(img_array, axis=0)
    pred_mask = unet_model.predict(img)[0, :, :, 0]
    return (pred_mask > 0.5).astype(np.uint8)

def overlay_mask_on_ir(image, mask):
    if image.ndim == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 3 and image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 2] = mask * 255  # Red channel for crack
    return cv2.addWeighted(image, 0.6, mask_colored, 0.4, 0)

def predict_ir_crack(img_array):
    mask = predict_ir_mask(img_array)
    crack_area = np.sum(mask)
    image_area = mask.shape[0] * mask.shape[1]
    crack_ratio = crack_area / image_area
    confidence = round(min(crack_ratio * 10, 1.0), 2)  # Clip to 1.0
    label = 1 if crack_ratio > 0.01 else 0
    return label, confidence
