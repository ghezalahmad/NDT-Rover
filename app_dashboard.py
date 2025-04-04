import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import plotly.express as px  # Import for plotting


# Set page configuration
st.set_page_config(page_title="Multi-Sensor Crack Detection", layout="wide")
st.title("NDT-Rover")


# --- Sidebar controls ---
with st.sidebar:
    st.header("Configuration")

    optical_path = st.text_input(
        "Optical Images Directory",
        value="DATA_Maguire_20180517_ALL/SDNET2018/D/CD"
    )

    ir_path = st.text_input(
        "Infrared Images Directory",
        value="Crack_detection/02-Infrared images"
    )

    optical_model_path = st.text_input(
        "Optical Model Path",
        value="optical_cnn_model.h5"
    )

    ir_model_path = st.text_input(
        "IR Model Path",
        value="ir_unet_model.h5"
    )

    sensor_selection = st.multiselect(
        "Select Sensors:",
        ["Optical Camera", "Infrared Camera"],
        default=["Optical Camera"]
    )

    st.session_state.confidence_threshold = st.slider("Minimum Confidence %", 0, 100, 50)

    if "Optical Camera" in sensor_selection:
        st.subheader("Optical Measurement Filters")
        st.session_state.min_crack_length = st.slider("Min Crack Length (pixels)", 10, 200, 50)
        st.session_state.min_crack_area = st.slider("Min Crack Area (pixels)", 5, 100, 20)
        st.session_state.min_aspect_ratio = st.slider("Min Aspect Ratio (L/W)", 1.0, 10.0, 3.0)
        st.session_state.max_aspect_ratio = st.slider("Max Aspect Ratio (L/W)", 0.1, 1.0, 0.3)
        st.session_state.blur_kernel_size = st.slider("Gaussian Blur Kernel Size (Odd, 0 for none)", 0, 15, 3, step=2)

    refresh_rate = st.slider("Refresh Rate (seconds)", 1, 100, 2)
    start_monitoring = st.button("Start Monitoring")
    stop_monitoring = st.button("Stop Monitoring")

# --- Initialize session state ---
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'optical_images' not in st.session_state:
    st.session_state.optical_images = []
if 'optical_infos' not in st.session_state:
    st.session_state.optical_infos = []
if 'infrared_images' not in st.session_state:
    st.session_state.infrared_images = []
if 'infrared_infos' not in st.session_state:
    st.session_state.infrared_infos = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'optical_model' not in st.session_state:
    st.session_state.optical_model = None
if 'ir_model' not in st.session_state:
    st.session_state.ir_model = None
if 'optical_crack_counts' not in st.session_state:
    st.session_state.optical_crack_counts = {}
# Add a counter for unique keys
if 'chart_counter' not in st.session_state:
    st.session_state.chart_counter = 0



# Update monitoring state based on button clicks
if start_monitoring:
    st.session_state.monitoring = True
if stop_monitoring:
    st.session_state.monitoring = False

# --- Model loading functions ---
def load_optical_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading optical model: {e}")
        return None

def load_ir_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading IR model: {e}")
        return None

def preprocess_image(path, mode="RGB", target_size=(128, 128)):
    """Process image from file path."""
    try:
        img = Image.open(path).convert(mode)
        resized_img = img.resize(target_size)
        arr = img_to_array(resized_img) / 255.0
        return arr, img
    except Exception as e:
        st.error(f"Error processing image {path}: {e}, type: {type(e)}")
        return None, None

import cv2  # Make sure this is at the top of your app-dashboard.py

def predict_optical_crack(image_array, original_image, model):
    """Predict crack using optical model and perform edge detection and basic measurement"""
    try:
        input_array = np.expand_dims(image_array, axis=0)
        pred = model.predict(input_array)[0]
        label = int(np.argmax(pred))
        conf = float(np.max(pred))

        visualized_image = original_image.copy()
        crack_measurements = []

        min_length = st.session_state.min_crack_length
        min_area = st.session_state.min_crack_area
        min_ar = st.session_state.min_aspect_ratio
        max_ar = st.session_state.max_aspect_ratio
        blur_kernel_size = st.session_state.blur_kernel_size

        visualized_image = original_image.copy()
        crack_measurements = []

        if label == 1:
            gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)

            # Apply Gaussian Blur if kernel size is greater than 0
            if blur_kernel_size > 0 and blur_kernel_size % 2 == 1:  # Kernel size should be positive and odd
                blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
                edges = cv2.Canny(blurred, 50, 150)
            else:
                edges = cv2.Canny(gray, 50, 150)

            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            visualized_image = cv2.addWeighted(np.array(original_image), 0.7, edges_rgb, 0.3, 0)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                length = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                if length > min_length and area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    if h > 0:
                        aspect_ratio = float(w) / h
                        if aspect_ratio > min_ar or aspect_ratio < max_ar:
                            crack_measurements.append({
                                "length": round(length, 1),
                                "width": min(w, h),
                                "aspect_ratio": round(aspect_ratio, 2),
                                "bbox": (x, y, w, h)
                            })
                            cv2.drawContours(np.array(visualized_image), [contour], -1, (0, 255, 0), 2)
                            cv2.putText(np.array(visualized_image), f'L:{round(length,1)}', (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            visualized_image = Image.fromarray(visualized_image)

        return label, conf, visualized_image, len(crack_measurements), crack_measurements # Return count
    except Exception as e:
        st.error(f"Error in optical prediction and measurement: {e}")
        return 0, 0.0, original_image, 0, []

def predict_ir_mask(image_array, model):
    """Predict mask using IR model"""
    try:
        input_array = np.expand_dims(image_array, axis=0)

        # Replicate the single channel to create a 3-channel input
        input_array_3_channel = np.repeat(input_array, 3, axis=-1)

        pred_mask = model.predict(input_array_3_channel)[0]

        if len(pred_mask.shape) == 2:
            pred_mask = np.expand_dims(pred_mask, axis=-1)
        elif len(pred_mask.shape) == 3 and pred_mask.shape[-1] != 1:
            pred_mask = np.expand_dims(pred_mask, axis=-1)

        pred_mask_thresholded = pred_mask > 0.2

        return pred_mask_thresholded.astype(np.float32)
    except Exception as e:
        st.error(f"Error in IR prediction: {e}, type: {type(e)}")
        return np.zeros((image_array.shape[0], image_array.shape[1], 1), dtype=np.float32)

def overlay_mask_on_ir(image, mask):
    """Overlay mask on IR image for visualization"""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    overlay = image.copy()
    if len(image.shape) == 2:
        overlay = np.stack([overlay] * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        overlay = np.concatenate([overlay] * 3, axis=2)

    display_mask = mask
    if len(display_mask.shape) > 2:
        display_mask = display_mask[:, :, 0]
    if len(display_mask.shape) == 1:
        display_mask = np.reshape(display_mask, (128, 128))

    overlay[:, :, 0] = np.where(display_mask > 0.5, 255, overlay[:, :, 0])
    overlay[:, :, 1] = np.where(display_mask > 0.5, 0, overlay[:, :, 1])
    overlay[:, :, 2] = np.where(display_mask > 0.5, 0, overlay[:, :, 2])

    result = image.copy()
    if len(result.shape) == 2:
        result = np.stack([result] * 3, axis=-1)
    elif len(result.shape) == 3 and result.shape[2] == 1:
        result = np.concatenate([result] * 3, axis=2)

    alpha = 0.5
    result = (result * (1 - alpha) + overlay * alpha).astype(np.uint8)

    return Image.fromarray(result)

# --- Define update_section function ---

optical_confidence_chart_placeholder = st.empty()

def update_section(sensor_key, container):
    with container.container():
        print(f"update_section called for {sensor_key}")
        st.markdown(f"### 🔹 {sensor_key.capitalize()} Camera")

        images_to_show = getattr(st.session_state, f"{sensor_key}_images")
        print(f'{sensor_key} images list length: {len(images_to_show)}')
        if images_to_show:
            cols = st.columns(4)
            for i, (img, caption) in enumerate(images_to_show[:8]):
                with cols[i % 4]:
                    st.image(img, caption=caption, use_container_width=True)
        else:
            st.info(f"No {sensor_key} images processed yet")

        infos = getattr(st.session_state, f"{sensor_key}_infos")
        print(f'{sensor_key} info list length: {len(infos)}')
        if sensor_key == "optical":
            if 'show_optical_confidence_chart' not in st.session_state:
                st.session_state.show_optical_confidence_chart = False

            if st.session_state.monitoring:
                if infos:
                    df = pd.DataFrame(infos)
                    if len(df) > 1 and "Confidence" in df.columns:
                        print("Attempting to create/update optical confidence chart")
                        # Increment the chart counter for a unique key
                        st.session_state.chart_counter += 1
                        unique_chart_key = f"optical_confidence_chart_{st.session_state.chart_counter}"

                        fig_conf = go.Figure()
                        fig_conf.add_trace(go.Scatter(x=df["Time"], y=df["Confidence"], mode="lines+markers"))
                        fig_conf.update_layout(title="Model Confidence Over Time", yaxis_title="Confidence (%)", xaxis_title="Time")
                        optical_confidence_chart_placeholder.plotly_chart(fig_conf, use_container_width=True, key=unique_chart_key)
                        st.session_state.show_optical_confidence_chart = True
                        st.dataframe(df)  # <--- ADDED THIS LINE TO DISPLAY THE TABLE
                    else:
                        st.session_state.show_optical_confidence_chart = False
                        print("Confidence data insufficient for optical chart")
                        if infos:
                            st.dataframe(pd.DataFrame(infos)) # Display table even if chart data is insufficient
                else:
                    st.session_state.show_optical_confidence_chart = False

            elif st.session_state.show_optical_confidence_chart and not st.session_state.monitoring and infos:
                df = pd.DataFrame(infos)
                if len(df) > 1 and "Confidence" in df.columns:
                    print("Rendering optical confidence chart one last time")
                    # Use another unique key for the final render
                    st.session_state.chart_counter += 1
                    final_chart_key = f"optical_confidence_chart_final_{st.session_state.chart_counter}"

                    fig_conf = go.Figure()
                    fig_conf.add_trace(go.Scatter(x=df["Time"], y=df["Confidence"], mode="lines+markers"))
                    fig_conf.update_layout(title="Model Confidence Over Time", yaxis_title="Confidence (%)", xaxis_title="Time")
                    optical_confidence_chart_placeholder.plotly_chart(fig_conf, use_container_width=True, key=final_chart_key)
                    st.dataframe(df) # <--- ADDED THIS LINE FOR THE FINAL RENDER TABLE
                else:
                    optical_confidence_chart_placeholder.empty()
                    st.session_state.show_optical_confidence_chart = False
                    if infos:
                        st.dataframe(pd.DataFrame(infos)) # Display table even if chart data is insufficient

            elif not st.session_state.monitoring:
                optical_confidence_chart_placeholder.empty()
                st.session_state.show_optical_confidence_chart = False
                if infos:
                    st.dataframe(pd.DataFrame(infos)) # Display table when monitoring is off


        if sensor_key == "optical" and st.session_state.optical_crack_counts:
            print("Attempting to create optical crack count chart")
            st.subheader("Detected Crack Count per Image")
            filenames = list(st.session_state.optical_crack_counts.keys())
            counts = list(st.session_state.optical_crack_counts.values())

            # Use a unique key for the crack count chart too
            st.session_state.chart_counter += 1
            crack_count_key = f"optical_crack_count_chart_{st.session_state.chart_counter}"

            fig_count = px.bar(x=filenames, y=counts, labels={'x':'Image File', 'y':'Crack Count'})
            st.plotly_chart(fig_count, use_container_width=True, key=crack_count_key)
        else:
            print("Optical crack count data insufficient for chart")

# --- Create placeholders for results ---
status_message = st.empty()
optical_section = st.empty()
infrared_section = st.empty()

# --- Load models ---
if "Optical Camera" in sensor_selection and st.session_state.optical_model is None:
    status_message.info("Loading optical model...")
    st.session_state.optical_model = load_optical_model(optical_model_path)

if "Infrared Camera" in sensor_selection and st.session_state.ir_model is None:
    status_message.info("Loading infrared model...")
    st.session_state.ir_model = load_ir_model(ir_model_path)

status_message.empty()

# --- Process files from directories ---
def process_optical_files():
    if not os.path.exists(optical_path):
        st.warning(f"Optical path not found: {optical_path}")
        return

    if st.session_state.optical_model is None:
        st.warning("Optical model not loaded")
        return

    files = sorted([f for f in os.listdir(optical_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                   and f not in st.session_state.processed_files])

    if not files:
        return

    # Process a limited number of files per refresh to avoid overload
    for fname in files[:5]:
        st.session_state.processed_files.add(fname)
        full_path = os.path.join(optical_path, fname)

        try:
            arr, img = preprocess_image(full_path)
            if arr is None:
                continue

            label, conf, visualized_img, crack_count, measurements = predict_optical_crack(arr, img, st.session_state.optical_model)

            st.session_state.optical_crack_counts[fname] = crack_count # Store the count

            if label == 1 and conf * 100 >= st.session_state.confidence_threshold:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.optical_images.insert(0, (visualized_img, f"{ts} | Crack | {conf*100:.1f}%"))
                measurement_strings = []
                if measurements:
                    for meas in measurements:
                        measurement_strings.append(f"L:{meas['length']:.1f}, W:{meas['width']}, AR:{meas['aspect_ratio']:.2f}")

                info = {
                    "Time": ts,
                    "File": fname,
                    "Confidence": round(conf * 100, 1),
                    "Status": "Crack"
                }
                if measurement_strings:
                    info["Measurements"] = ", ".join(measurement_strings)
                st.session_state.optical_infos.insert(0, info)
        except Exception as e:
            st.error(f"Error processing {fname}: {e}")

def process_ir_files():
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
        return

    for fname in files[:5]:
        st.session_state.processed_files.add(fname)
        full_path = os.path.join(ir_path, fname)

        try:
            arr, img = preprocess_image(full_path, mode="L") # Load IR images as grayscale
            if arr is None:
                print(f"Image preprocessing failed for {fname}")
                continue

            mask = predict_ir_mask(arr, st.session_state.ir_model)

            crack_area = np.sum(mask) / (mask.shape[0] * mask.shape[1])
            confidence = crack_area * 100
            label = 1 if crack_area > 0.005 else 0

            print(f"IR File: {fname}, Crack Area: {crack_area:.4f}, Confidence: {confidence:.2f}%, Label: {label}, Threshold: {st.session_state.confidence_threshold}") # Debugging

            if label == 1 and confidence >= st.session_state.confidence_threshold:
                print(f"DEBUG: Condition met for {fname} - Label: {label}, Confidence: {confidence}, Threshold: {st.session_state.confidence_threshold}") # Debugging
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                overlay = overlay_mask_on_ir((arr * 255).astype(np.uint8), mask)
                st.session_state.infrared_images.insert(0, (overlay, f"{ts} | Crack | {confidence:.1f}%"))
                st.session_state.infrared_infos.insert(0, {
                    "Time": ts,
                    "File": fname,
                    "Confidence": round(confidence, 1),
                    "Status": "Crack"
                })
                print(f"Infrared image added: {fname}")
            else:
                print(f"Image {fname} filtered out.")

        except Exception as e:
            st.error(f"Error processing {fname}: {e}, type: {type(e)}")

# --- Processing and Display Loop ---
while st.session_state.monitoring:
    # Process files
    if "Optical Camera" in sensor_selection:
        process_optical_files()
        update_section("optical", optical_section)

    if "Infrared Camera" in sensor_selection:
        process_ir_files()
        update_section("infrared", infrared_section)

    time.sleep(refresh_rate)

# --- Always update the display one last time when monitoring stops ---
if "Optical Camera" in sensor_selection:
    update_section("optical", optical_section)

if "Infrared Camera" in sensor_selection:
    update_section("infrared", infrared_section)

# Add a monitoring status indicator
st.sidebar.markdown("---")
if st.session_state.monitoring:
    st.sidebar.success("Monitoring Active")
else:
    st.sidebar.warning("Monitoring Paused")

# Add a footer
st.markdown("---")
st.markdown("*This application performs real-time crack detection using optical and infrared imagery.*")