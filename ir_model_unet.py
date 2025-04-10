# ir_model_unet.py — U-Net Inference for Infrared Crack Segmentation with Streamlit Integration

import os
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import streamlit as st
import cv2
from tensorflow.keras.models import load_model

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
    
    return Image.fromarray(rgb_image)



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
        
        if confidence >= st.session_state.confidence_threshold:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            try:
                overlay = overlay_mask_on_ir(img, mask)
                # Ensure that the overlay image is added to the infrared images list
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





import plotly.graph_objects as go

def update_ir_section(container):
    with container.container():
        st.markdown("### 🔹 Infrared Camera")

        # Display the infrared images
        if st.session_state.infrared_images:
            cols = st.columns(4)
            for i, (img, caption) in enumerate(st.session_state.infrared_images[:8]):
                with cols[i % 4]:
                    st.image(img, caption=caption, use_container_width=True)
        else:
            st.info("No infrared images processed yet")

        # Display the data in table form
        if st.session_state.infrared_infos:
            df = pd.DataFrame(st.session_state.infrared_infos)
            st.dataframe(df)

        # Plot the distribution of crack confidence
        if st.session_state.infrared_infos:
            confidence_values = [entry['Confidence'] for entry in st.session_state.infrared_infos]
            
            fig = go.Figure()

            # Create the histogram of confidence values
            fig.add_trace(go.Histogram(
                x=confidence_values,
                nbinsx=20,
                name="Crack Confidence",
                marker=dict(color="rgba(255, 99, 132, 0.6)"),
            ))

            fig.update_layout(
                title="Distribution of Crack Confidence",
                xaxis_title="Confidence (%)",
                yaxis_title="Frequency",
                showlegend=False
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
