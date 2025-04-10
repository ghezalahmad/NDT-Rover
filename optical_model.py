import os
import time
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2
import pandas as pd
import datetime
import plotly.graph_objects as go

def load_image(path, target_size=(128, 128)):
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
    """Uses U-Net model to generate a segmentation mask for cracks"""
    if model is None:
        st.warning("Optical model not loaded.")
        return None
    try:
        # For U-Net segmentation model
        prediction = model.predict(img_array)
        print(f"Debug (detect_cracks): Raw prediction shape: {prediction.shape}, dtype: {prediction.dtype}")
        
        # Ensure the prediction is reshaped to a 2D mask
        if len(prediction.shape) == 4:  # Example: (1, height, width, channels)
            prediction = prediction[0, :, :, 0]  # Extract the first channel
        elif len(prediction.shape) == 3:  # Example: (height, width, channels)
            prediction = prediction[:, :, 0]  # Extract the first channel
        elif len(prediction.shape) != 2:  # Invalid shape
            print("Debug (detect_cracks): Invalid prediction shape, skipping.")
            return None

        # Apply threshold to get binary mask
        binary_mask = prediction > threshold
        
        # Debug: Check mask content
        print(f"Debug (detect_cracks): Binary mask shape: {binary_mask.shape}, non-zero pixels: {np.sum(binary_mask)}")

        # Check if any cracks are detected
        if np.any(binary_mask):
            print("Debug (detect_cracks): Crack detected in segmentation mask.")
            return binary_mask
        else:
            print("Debug (detect_cracks): No crack detected in segmentation mask.")
            return None

    except Exception as e:
        st.error(f"Error during optical prediction: {e}")
        return None

def measure_crack(mask):
    mask_np = mask.astype(np.uint8) * 255
    print(f"Debug: Measuring crack on mask, shape: {mask_np.shape}, dtype: {mask_np.dtype}")
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        (x, y), (w, h), angle = rect
        area = cv2.contourArea(largest_contour)
        length = max(w, h)
        width = min(w, h)
        aspect_ratio = length / width if width > 0 else 0
        print(f"Debug: Measured crack - Length: {length}, Width: {width}, Area: {area}, Aspect Ratio: {aspect_ratio}")
        return length, width, area, aspect_ratio
    else:
        print("Debug: No contours found in mask for measurement.")
        return None, None, None, None

def overlay_mask(image, mask):
    """Overlay the binary mask on the original image with red color only on detected cracks"""
    # Convert image to numpy array if it's not already
    overlay = np.array(image.copy())
    
    # Ensure mask is the right size and type
    if mask is None:
        print("Debug: Mask is None, returning original image")
        return Image.fromarray(overlay)
    
    # Convert mask to binary and proper dimensions
    if mask.shape[:2] != (overlay.shape[0], overlay.shape[1]):
        print(f"Debug: Resizing mask from {mask.shape} to {overlay.shape[:2]}")
        mask_resized = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]))
    else:
        mask_resized = mask.astype(np.uint8)
    
    # Create mask with 3 channels for RGB
    mask_rgb = np.zeros_like(overlay)
    
    # Set the red channel only where mask is positive
    mask_rgb[mask_resized > 0, 2] = 255  # Red channel (BGR format)
    
    # Print debug information
    non_zero_pixels = np.sum(mask_resized > 0)
    print(f"Debug: Mask has {non_zero_pixels} non-zero pixels out of {mask_resized.size}")
    
    if non_zero_pixels == 0:
        print("Debug: No pixels to overlay, returning original image")
        return Image.fromarray(overlay)
    
    if non_zero_pixels == mask_resized.size:
        print("Debug: WARNING - Entire mask is non-zero!")
    
    # Create a blend of original image and red mask only where mask is positive
    alpha = 0.5  # Transparency factor
    blend = cv2.addWeighted(overlay, 1, mask_rgb, alpha, 0)
    
    return Image.fromarray(blend)

def process_optical_files(optical_path, container):
    if not os.path.exists(optical_path):
        st.warning(f"Optical path not found: {optical_path}")
        return

    if st.session_state.optical_model is None:
        st.warning("Optical model not loaded.")
        return

    # Get unprocessed files
    files = sorted([f for f in os.listdir(optical_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                   and f not in st.session_state.processed_optical_files])

    print(f"Debug: Found optical files: {files[:5]}")

    for fname in files[:5]:
        st.session_state.processed_optical_files.add(fname)  # Mark file as processed
        full_path = os.path.join(optical_path, fname)
        print(f"Debug: Processing optical file: {full_path}")

        original_img_resized, original_img = load_image(full_path)
        if original_img_resized is None:
            continue

        img_array = preprocess_image(original_img_resized)
        crack_mask = detect_cracks(st.session_state.optical_model, img_array, threshold=st.session_state.confidence_threshold / 100.0)

        if crack_mask is not None and np.any(crack_mask):
            print("Debug: Crack detected in image.")
            try:
                # Apply blur if specified
                if st.session_state.blur_kernel_size > 0 and st.session_state.blur_kernel_size % 2 == 1:
                    blur_kernel = st.session_state.blur_kernel_size
                    mask_np = crack_mask.astype(np.uint8) * 255
                    blurred_mask_np = cv2.GaussianBlur(mask_np, (blur_kernel, blur_kernel), 0)
                    crack_mask_processed = blurred_mask_np > 127
                else:
                    crack_mask_processed = crack_mask.astype(np.uint8)

                # Measure crack properties
                length, width, area, aspect_ratio = measure_crack(crack_mask_processed)

                # Debugging crack measurements
                print(f"Debug: Crack measurements - Length: {length}, Width: {width}, Area: {area}, Aspect Ratio: {aspect_ratio}")
                print(f"Debug: Filtering criteria - Min Length: {st.session_state.min_crack_length}, Min Area: {st.session_state.min_crack_area}, "
                      f"Min Aspect Ratio: {st.session_state.min_aspect_ratio}, Max Aspect Ratio: {st.session_state.max_aspect_ratio}")

                # Skip invalid cracks
                if length is None or width is None or area is None or aspect_ratio is None:
                    print("Debug: Invalid crack measurements, skipping overlay.")
                    continue

                # Filter based on size criteria
                if (length >= st.session_state.min_crack_length and
                        area >= st.session_state.min_crack_area and
                        st.session_state.min_aspect_ratio <= aspect_ratio <= st.session_state.max_aspect_ratio):
                    
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    overlaid_image = overlay_mask(original_img, crack_mask_processed)
                    confidence = np.mean(crack_mask) * 100

                    st.session_state.optical_images.insert(0, (overlaid_image, f"{ts} | Crack | {confidence:.1f}%"))
                    st.session_state.optical_infos.insert(0, {
                        "Time": ts,
                        "File": fname,
                        "Length (pixels)": round(length, 2),
                        "Width (pixels)": round(width, 2),
                        "Area (pixels)": round(area, 2),
                        "Aspect Ratio (L/W)": round(aspect_ratio, 2),
                        "Confidence": round(confidence, 1)
                    })
                    print(f"Debug: Crack added to session state for debugging.")
                else:
                    print("Debug: Crack did not meet filtering criteria.")
            except Exception as e:
                st.error(f"Error processing mask: {e}")
                continue
        else:
            print("Debug: No crack detected or mask doesn't meet criteria.")

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

            # Add a plot for crack properties
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df["Time"],
                y=df["Length (pixels)"],
                name="Length"
            ))
            fig.add_trace(go.Bar(
                x=df["Time"],
                y=df["Width (pixels)"],
                name="Width"
            ))
            fig.add_trace(go.Bar(
                x=df["Time"],
                y=df["Area (pixels)"],
                name="Area"
            ))
            fig.update_layout(
                title="Crack Properties Over Time",
                xaxis_title="Time",
                yaxis_title="Pixels",
                barmode="group"
            )
            # Generate a unique key using the current timestamp
            unique_key = f"optical_crack_plot_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            st.plotly_chart(fig, key=unique_key)