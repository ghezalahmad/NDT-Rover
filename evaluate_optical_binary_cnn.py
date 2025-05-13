import os
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# --- Config ---
CD_PATH = "DATA_Maguire_20180517_ALL/SDNET2018/D/CD"
UD_PATH = "DATA_Maguire_20180517_ALL/SDNET2018/D/UD"
IMG_SIZE = (128, 128)
MODEL_PATH = "optical_binary_classifier.h5"  # must match your training output

# --- Load Images (Same as Training) ---
def load_data(cd_path, ud_path, img_size=(128, 128)):
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

# --- Evaluate ---
if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data(CD_PATH, UD_PATH, IMG_SIZE)
    
    # Same split method used during training
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Running predictions...")
    y_pred = (model.predict(X_val) > 0.5).astype(int).flatten()

    print("\n--- Evaluation Report ---")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print("Classification Report:\n", classification_report(y_val, y_pred, target_names=["No Crack", "Crack"]))
