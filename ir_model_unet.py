# ir_model_unet.py — U-Net Inference for Infrared Crack Segmentation with Streamlit Integration

import os
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops


def preprocess_image(path, target_size=(128, 128)):
    try:
        img = Image.open(path).convert("L")  # Convert to grayscale
        resized_img = img.resize(target_size)
        # Convert grayscale to 3-channel format expected by the model
        arr = np.array(resized_img) / 255.0
        # Add the channel dimension to match expected input shape (height, width, channels)
        arr = np.expand_dims(arr, axis=-1)
        # Repeat the grayscale channel 3 times to create RGB
        arr = np.repeat(arr, 3, axis=-1)
        print(f"Debug: Image shape after conversion: {arr.shape}")  # Should be (128, 128, 3)

        return arr, img
    except Exception as e:
        st.error(f"Error processing image {path}: {e}")
        return None, None

def predict_ir_mask(image_array, model):
    try:
        input_array = np.expand_dims(image_array, axis=0)
        pred_mask = model.predict(input_array)[0]
        if len(pred_mask.shape) == 3 and pred_mask.shape[2] > 1:
            pred_mask = pred_mask[:, :, 0]  # Take the first channel if needed
            
        # Ensure the prediction is a 2D array and binary
        return (pred_mask > 0.2).astype(np.uint8)  # Changed to uint8 for compatibility
    except Exception as e:
        st.error(f"Error in IR prediction: {e}")
        print(f"Debug: Input array shape was {image_array.shape}")
        return np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)


def overlay_mask_on_ir(image, mask):
    """Overlay mask on the original image - highlight only the cracks"""
    # Convert original image to numpy array
    original = np.array(image).copy()
    
    # Make sure mask is properly sized for the image
    if mask.shape[:2] != original.shape[:2]:
        # Use cv2 instead of PIL for resizing the mask
        mask_resized = cv2.resize(
            mask.astype(np.uint8), 
            (original.shape[1], original.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        mask = mask_resized > 0
    
    # Convert grayscale images to RGB for better visualization in Streamlit
    if len(original.shape) == 2:  # If it's a grayscale image
        rgb_image = np.stack([original] * 3, axis=-1)
        rgb_image[mask] = [255, 0, 0]  # Red color for cracks
    else:  # If it's an already RGB image
        rgb_image = original.copy()
        rgb_image[mask] = [255, 0, 0]  # Red color for cracks
    
    return Image.fromarray(rgb_image.astype(np.uint8)).convert("RGB")




def process_ir_files(ir_path, container):
    if not os.path.exists(ir_path):
        st.warning(f"IR path not found: {ir_path}")
        return

    if st.session_state.ir_model is None:
        st.warning("IR model not loaded")
        return

    files = sorted([f for f in os.listdir(ir_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                   and f not in st.session_state.processed_files])

    if not files:
        st.warning("No new infrared images to process.")
        return

    st.success(f"Processing {len(files)} infrared images")

    for fname in files[:5]:  # Process up to 5 images
        st.session_state.processed_files.add(fname)
        full_path = os.path.join(ir_path, fname)
        print(f"Debug: Processing IR file: {full_path}")

        arr, img = preprocess_image(full_path)
        if arr is None:
            continue

        mask = predict_ir_mask(arr, st.session_state.ir_model)

        print(f"Debug: Mask after prediction: shape={mask.shape}, dtype={mask.dtype}, " 
              f"positive pixels={np.sum(mask)}, total={mask.size}")
        
        confidence = np.sum(mask) / mask.size * 100
        print(f"Debug: Confidence for {fname}: {confidence}%")
        
        # Calculate additional crack features
        crack_area = calculate_crack_area(mask)
        crack_length = calculate_crack_length(mask)
        crack_width = calculate_crack_width(mask)
        # Calculate texture features
        contrast, correlation = calculate_texture_features(mask)
        st.session_state.texture_contrast.append(float(contrast[0][0]))
        st.session_state.texture_correlation.append(float(correlation[0][0]))

        
        print(f"Debug: Crack Area: {crack_area}, Crack Length: {crack_length}, Crack Width: {crack_width}")

        # Store features in session state
        st.session_state.crack_areas.append(crack_area)
        st.session_state.crack_lengths.append(crack_length)
        st.session_state.crack_widths.append(crack_width)

        if confidence >= st.session_state.confidence_threshold:
            # Bypass threshold check during debugging
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            try:
                overlay = overlay_mask_on_ir(img, mask)
                print(f"Overlay type: {type(overlay)}")
                st.session_state.infrared_images.insert(0, (overlay, f"{ts} | Crack | {confidence:.1f}%"))
                st.session_state.infrared_infos.insert(0, {
                    "Time": ts,
                    "File": fname,
                    "Confidence": round(confidence, 1),
                    "Status": "Crack"
                })
            except Exception as e:
                st.error(f"Error creating overlay: {e}")
                import traceback
                print(traceback.format_exc())





def calculate_crack_area(mask):
    return np.sum(mask)  # Count non-zero pixels (crack area)

def calculate_crack_length(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crack_length = 0
    for contour in contours:
        crack_length += cv2.arcLength(contour, closed=False)
    return crack_length

def calculate_crack_width(mask):
    width_values = []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        for point in contour:
            width_values.append(np.linalg.norm(point[0] - contour[0][0]))
    return np.mean(width_values) if width_values else 0

def get_crack_position(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    positions = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            positions.append((cX, cY))
    return positions

def calculate_texture_features(image):
    if len(image.shape) == 3:
        image = image[:, :, 0]

    image = (image * 255).astype(np.uint8)  # Scale to 0–255
    glcm = graycomatrix(image, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    correlation = graycoprops(glcm, 'correlation')
    return contrast, correlation




import plotly.graph_objects as go
def update_ir_section(container):
    with container.container():
        st.markdown("### 🔹 Infrared Camera")

        # --- Image Display Section ---
        with st.expander("📷 View Infrared Crack Images", expanded=True):
            if st.session_state.infrared_images:
                cols = st.columns(4)
                for i, (img, caption) in enumerate(st.session_state.infrared_images[:8]):
                    with cols[i % 4]:
                        st.image(img, caption=caption, use_container_width=True)
            else:
                st.info("No infrared images processed yet")

        # --- Table Section ---
        with st.expander("📋 Crack Detection Table", expanded=True):
            if st.session_state.infrared_infos:
                df = pd.DataFrame(st.session_state.infrared_infos)
                st.dataframe(df)
            else:
                st.info("No crack info table to display")

        # --- Crack Distribution Charts ---
        with st.expander("📊 Crack Feature Distributions"):
            if st.session_state.crack_areas:
                st.plotly_chart(go.Figure(
                    go.Histogram(x=st.session_state.crack_areas, name="Crack Area", opacity=0.75)
                ).update_layout(
                    title="Crack Area Distribution", xaxis_title="Area (px)", yaxis_title="Count", bargap=0.2
                ), use_container_width=True)

            if st.session_state.crack_lengths:
                st.plotly_chart(go.Figure(
                    go.Histogram(x=st.session_state.crack_lengths, name="Crack Length", opacity=0.75)
                ).update_layout(
                    title="Crack Length Distribution", xaxis_title="Length (px)", yaxis_title="Count", bargap=0.2
                ), use_container_width=True)

            if st.session_state.crack_widths:
                st.plotly_chart(go.Figure(
                    go.Histogram(x=st.session_state.crack_widths, name="Crack Width", opacity=0.75)
                ).update_layout(
                    title="Crack Width Distribution", xaxis_title="Width (px)", yaxis_title="Count", bargap=0.2
                ), use_container_width=True)

        # --- Crack Trends Over Time ---
        with st.expander("📈 Crack Feature Trends (Last 30 Images)", expanded=False):
            if st.session_state.crack_areas:
                trend_df = pd.DataFrame({
                    "Area": st.session_state.crack_areas[-30:],
                    "Length": st.session_state.crack_lengths[-30:],
                    "Width": st.session_state.crack_widths[-30:]
                })
                st.line_chart(trend_df)
            else:
                st.info("No trend data yet.")

        # --- Texture Features (GLCM Contrast & Correlation) ---
        with st.expander("🧠 Texture Features (Contrast / Correlation)", expanded=False):
            if "texture_contrast" in st.session_state and "texture_correlation" in st.session_state:
                tex_df = pd.DataFrame({
                    "Contrast": st.session_state.texture_contrast[-30:],
                    "Correlation": st.session_state.texture_correlation[-30:]
                })
                st.dataframe(tex_df)

                st.line_chart(tex_df)
            else:
                st.info("Texture features will appear as new cracks are processed.")

