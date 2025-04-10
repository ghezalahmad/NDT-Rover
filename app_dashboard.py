# Refactored version of app_dashboard.py with improved monitoring loop,
# global unified log, and safer session state handling.

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import datetime
import cv2
import joblib


from utils import load_model_safe, load_gas_model, load_csv_with_timestamp
from gas_simulator import run_gas_simulator
from optical_model import process_optical_files, update_optical_section
from ir_model_unet import process_ir_files, update_ir_section

# --- Page Configuration ---
st.set_page_config(page_title="Multi-Sensor Crack Detection", layout="wide")
st.title("NDT-Rover")

# --- Sidebar Configuration ---
# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")

    gas_path = st.text_input("Gas Data Directory", value="gas_data/MQ135SensorData.csv")
    gas_model_path = st.text_input("Gas Model Path", value="gas_model.pkl")  # Update this line
    gas_scaler_path = st.text_input("Gas Scaler Path", value="gas_scaler.pkl")  # Add this line

    optical_path = st.text_input("Optical Images Directory", value="DATA_Maguire_20180517_ALL/SDNET2018/D/CD")
    ir_path = st.text_input("Infrared Images Directory", value="Crack_detection/02-Infrared images")
    optical_model_path = st.text_input("Optical Model Path", value="optical_cnn_model.h5")
    ir_model_path = st.text_input("IR Model Path", value="ir_unet_model.h5")

    sensor_selection = st.multiselect(
        "Select Sensors:",
        ["Optical Camera", "Infrared Camera", "Gas Detector"],
        default=["Optical Camera"]
    )

    st.session_state.confidence_threshold = st.slider("Minimum Confidence %", 0, 100, 50)

    if "Optical Camera" in sensor_selection:
        st.subheader("Optical Measurement Filters")
        st.session_state.min_crack_length = st.slider("Min Crack Length (pixels)", 1, 200, 1)
        st.session_state.min_crack_area = st.slider("Min Crack Area (pixels)", 0, 100, 0)
        st.session_state.min_aspect_ratio = st.slider("Min Aspect Ratio (L/W)", 0.0, 10.0, 0.0)
        st.session_state.max_aspect_ratio = st.slider("Max Aspect Ratio (L/W)", 0.0, 10.0, 10.0)
        st.session_state.blur_kernel_size = st.slider("Gaussian Blur Kernel Size (Odd, 0 for none)", 0, 15, 3, step=2)

    start_monitoring = st.button("Start Monitoring")
    stop_monitoring = st.button("Stop Monitoring")


# --- Initialize Session State Keys Safely ---
def init_session():
    defaults = {
        'monitoring': False,
        'optical_images': [],
        'optical_infos': [],
        'infrared_images': [],
        'infrared_infos': [],
        'gas_data_history': pd.DataFrame(),
        'gas_alerts': [],
        'processed_optical_files': set(),
        'processed_ir_files': set(),
        'processed_files': set(),
        'gas_df': pd.DataFrame(),
        'gas_index': 0,
        'optical_model': None,
        'ir_model': None,
        'gas_model': None,
        'monitoring_log': [],
        'crack_areas': [],    # Initialize crack_areas here
        'crack_lengths': [],  # Initialize crack_lengths here
        'crack_widths': [],   # Initialize crack_widths here
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_session()




# --- Handle Monitoring Toggle ---
if start_monitoring:
    st.session_state.monitoring = True
if stop_monitoring:
    st.session_state.monitoring = False

# --- Load Models if Not Already Loaded ---
# --- Load Models if Not Already Loaded ---
if "Optical Camera" in sensor_selection and st.session_state.optical_model is None:
    st.session_state.optical_model = load_model_safe(optical_model_path)
if "Infrared Camera" in sensor_selection and st.session_state.ir_model is None:
    st.session_state.ir_model = load_model_safe(ir_model_path)

if "Gas Detector" in sensor_selection and st.session_state.gas_model is None:
    st.session_state.gas_model = load_gas_model(gas_model_path)
    if st.session_state.gas_model is None:
        st.error(f"Failed to load gas model from {gas_model_path}")
    
    # Load the gas data
    st.session_state.gas_df = load_csv_with_timestamp(gas_path)
    
    # Debugging: Check if gas data is loaded correctly
    print(f"Gas Data Path: {gas_path}")
    print(f"Gas Data (first 5 rows):\n{st.session_state.gas_df.head()}")
    
    # Initialize gas_index
    st.session_state.gas_index = 0



# --- Create Placeholders ---
status_message = st.empty()
optical_section = st.empty()
ir_section = st.empty()
gas_section = st.empty()

# --- Monitoring Execution ---
if st.session_state.monitoring:
    while st.session_state.monitoring:
        if "Optical Camera" in sensor_selection:
            process_optical_files(optical_path, optical_section)
            update_optical_section(optical_section)

        if "Infrared Camera" in sensor_selection:
            process_ir_files(ir_path, ir_section)
            update_ir_section(ir_section)

        if "Gas Detector" in sensor_selection:
            run_gas_simulator(st.session_state.gas_df, st.session_state.gas_model, gas_section)

        # Sleep for 2 seconds before processing the next batch
        time.sleep(2)

# --- Footer ---
st.sidebar.markdown("---")
if st.session_state.monitoring:
    st.sidebar.success("Monitoring Active")
else:
    st.sidebar.warning("Monitoring Paused")

# --- Log Download Option ---
if st.session_state.monitoring_log:
    df_log = pd.DataFrame(st.session_state.monitoring_log)
    st.sidebar.download_button("📥 Download Monitoring Log", df_log.to_csv(index=False), "monitoring_log.csv")

st.markdown("---")
st.markdown("*This application performs real-time crack detection using optical and infrared imagery along with gas detection alerts.*")
st.markdown("*For more information, please refer to the documentation.*")
