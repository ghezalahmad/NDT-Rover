import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_images_from_directory(base_path, label_map, img_size=(128, 128), color_mode="grayscale"):
    X, y = [], []
    for label_name, label_value in label_map.items():
        folder = os.path.join(base_path, label_name)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if color_mode == "grayscale" else cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    X.append(img)
                    y.append(label_value)
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)
    if color_mode == "grayscale":
        X = np.expand_dims(X, axis=-1)
    return X, y

def load_sdnet2018_images(root_path, label_map, img_size=(128, 128)):
    X, y = [], []
    for main_dir in ["D", "P", "W"]:
        full_main_path = os.path.join(root_path, main_dir)
        if not os.path.isdir(full_main_path):
            continue
        for subfolder in os.listdir(full_main_path):
            if subfolder not in label_map:
                continue
            folder_path = os.path.join(full_main_path, subfolder)
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, fname)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        X.append(img / 255.0)
                        y.append(label_map[subfolder])
    return np.array(X), np.array(y)


def prepare_dataset(X, y, test_size=0.2, random_state=42):
    y_cat = to_categorical(y)
    return train_test_split(X, y_cat, test_size=test_size, random_state=random_state)

def get_label_map(dataset_type="optical"):
    if dataset_type == "optical":
        return {"CD": 1, "UD": 0}

