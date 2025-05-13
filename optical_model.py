# Rewriting the fixed MobileNetV2-compatible optical_model.py after kernel reset


import os
import time
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import plotly.graph_objects as go

def load_image(path, target_size=(160, 160)):
    try:
        img = Image.open(path).convert('RGB')
        resized_img = img.resize(target_size)
        print(f"Debug: Loaded and resized image: {path}, size: {resized_img.size}")
        return resized_img, img
    except Exception as e:
        st.error(f"Error loading image {path}: {e}")
        return None, None

def preprocess_image(img):
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    print(f"Debug: Preprocessed image array, shape: {img_array.shape}, dtype: {img_array.dtype}")
    return img_array

def detect_cracks(model, img_array, threshold=0.5):
    if model is None:
        st.warning("Optical model not loaded.")
        return None, 0.0
    try:
        pred = model.predict(img_array)[0][0]
        print(f"Debug: Model prediction confidence: {pred}")
        detected = pred > threshold
        return detected, pred
    except Exception as e:
        st.error(f"Error during optical prediction: {e}")
        return None, 0.0

def process_optical_files(optical_path, container):
    if not os.path.exists(optical_path):
        st.warning(f"Optical path not found: {optical_path}")
        return

    if st.session_state.optical_model is None:
        st.warning("Optical model not loaded.")
        return

    files = sorted([f for f in os.listdir(optical_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                   and f not in st.session_state.processed_optical_files])

    print(f"Debug: Found optical files: {files[:5]}")

    for fname in files[:5]:
        st.session_state.processed_optical_files.add(fname)
        full_path = os.path.join(optical_path, fname)
        print(f"Debug: Processing optical file: {full_path}")

        original_img_resized, original_img = load_image(full_path)
        if original_img_resized is None:
            continue

        img_array = preprocess_image(original_img_resized)
        detected, confidence = detect_cracks(
            st.session_state.optical_model,
            img_array,
            threshold=st.session_state.confidence_threshold / 100.0
        )

        if detected:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.optical_images.insert(0, (original_img, f"{ts} | Crack | {confidence:.2f}"))
            st.session_state.optical_infos.insert(0, {
                "Time": ts,
                "File": fname,
                "Confidence": round(confidence, 2)
            })
            print("Debug: Crack classified and added to dashboard.")
        else:
            print("Debug: No crack classified.")

def update_optical_section(container):
    with container.container():
        st.markdown("### 📸 Optical Camera")
        print(f"Debug: Displaying optical images: {len(st.session_state.optical_images)}")
        print(f"Debug: Displaying optical infos: {len(st.session_state.optical_infos)}")
        if st.session_state.optical_images:
            cols = st.columns(4)
            for i, (img, caption) in enumerate(st.session_state.optical_images[:8]):
                with cols[i % 4]:
                    st.image(img, caption=caption, use_container_width=True)
        else:
            st.info("No optical images processed yet.")

        if st.session_state.optical_infos:
            df = pd.DataFrame(st.session_state.optical_infos)
            st.dataframe(df)

