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

# --- Initialize Session State Keys Safely ---
def init_session():
    if 'start_time' not in st.session_state:
        st.session_state['start_time'] = time.time()
        
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
        'crack_areas': [],
        'crack_lengths': [],
        'crack_widths': [],
        'confidence_threshold': 50,
        'min_crack_length': 1,
        'min_crack_area': 0,
        'min_aspect_ratio': 0.0,
        'max_aspect_ratio': 10.0,
        'blur_kernel_size': 3,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

# Initialize session state FIRST before accessing any values
init_session()

# For debugging: reset processed IR files
st.session_state['processed_files'].clear()
st.session_state['infrared_images'].clear()
st.session_state['infrared_infos'].clear()


# --- Page Configuration ---
st.set_page_config(
    page_title="NDT-Rover",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #3498db;
    }
    .subheader {
        font-size: 1.5rem;
        color: #3498db;
        margin-top: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .sensor-icon {
        font-size: 1.2rem;
        margin-right: 0.5rem;
    }
    .status-active {
        color: #2ecc71;
        font-weight: bold;
    }
    .status-inactive {
        color: #e74c3c;
        font-weight: bold;
    }
    .sidebar .css-1d391kg {
        padding-top: 2rem;
    }
    footer {
        text-align: center;
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        color: #7f8c8d;
        font-size: 0.8rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# --- Improved Header ---
st.markdown('<h1 class="main-header">NDT-Rover Monitoring System</h1>', unsafe_allow_html=True)
st.markdown('Multi-sensor crack detection and analysis system for infrastructure monitoring')

# --- Dashboard Metrics ---
metric_cols = st.columns(4)

with metric_cols[0]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{st.session_state["confidence_threshold"]}%</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Confidence Threshold</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with metric_cols[1]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    files_count = len(st.session_state['processed_optical_files']) + len(st.session_state['processed_ir_files'])
    st.markdown(f'<div class="metric-value">{files_count}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Processed Files</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with metric_cols[2]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    crack_count = len(st.session_state['crack_areas'])
    st.markdown(f'<div class="metric-value">{crack_count}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Detected Cracks</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with metric_cols[3]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    alert_count = len(st.session_state['gas_alerts'])
    st.markdown(f'<div class="metric-value">{alert_count}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Gas Alerts</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    # Add a placeholder for logo
    st.image("https://via.placeholder.com/150x80?text=NDT+Rover", width=150)  
    st.markdown("### 🔧 Configuration")

    # Create expandable sections for better organization
    with st.expander("📁 Data Directories", expanded=True):
        gas_path = st.text_input("Gas Data CSV", value="gas_data/MQ135SensorData.csv")
        optical_path = st.text_input("Optical Images", value="DATA_Maguire_20180517_ALL/SDNET2018/D/CD")
        ir_path = st.text_input("Infrared Images", value="Crack_detection/02-Infrared images")
    
    with st.expander("🧠 Model Paths", expanded=False):
        gas_model_path = st.text_input("Gas Model", value="gas_model.pkl")
        gas_scaler_path = st.text_input("Gas Scaler", value="gas_scaler.pkl")
        optical_model_path = st.text_input("Optical Model", value="optical_cnn_model.h5")
        ir_model_path = st.text_input("IR Model", value="ir_unet_model.h5")

    st.markdown("### 📊 Sensor Selection")
    sensor_selection = st.multiselect(
        "Select Sensors:",
        ["Optical Camera", "Infrared Camera", "Gas Detector"],
        default=["Optical Camera"]
    )

    st.markdown("### 🎚️ Detection Settings")
    st.session_state['confidence_threshold'] = st.slider("Confidence Threshold (%)", 0, 100, 
                                                        st.session_state['confidence_threshold'])

    # Conditional settings based on selected sensors
    if "Optical Camera" in sensor_selection:
        with st.expander("Optical Measurement Filters", expanded=False):
            st.session_state['min_crack_length'] = st.slider("Min Crack Length (px)", 1, 200, 
                                                            st.session_state['min_crack_length'])
            st.session_state['min_crack_area'] = st.slider("Min Crack Area (px)", 0, 100, 
                                                          st.session_state['min_crack_area'])
            st.session_state['min_aspect_ratio'] = st.slider("Min Aspect Ratio", 0.0, 10.0, 
                                                            st.session_state['min_aspect_ratio'])
            st.session_state['max_aspect_ratio'] = st.slider("Max Aspect Ratio", 0.0, 10.0, 
                                                            st.session_state['max_aspect_ratio'])
            st.session_state['blur_kernel_size'] = st.slider("Blur Kernel Size", 0, 15, 
                                                            st.session_state['blur_kernel_size'], step=2)

    # Control buttons with colors
    col1, col2 = st.columns(2)
    with col1:
        start_monitoring = st.button("▶️ Start", use_container_width=True, type="primary")
    with col2:
        stop_monitoring = st.button("⏹️ Stop", use_container_width=True)

    # Status indicator in sidebar
    st.markdown("---")
    if st.session_state['monitoring']:
        st.success("🟢 Monitoring Active")
    else:
        st.warning("🔴 Monitoring Paused")

    # Log Download Option
    if st.session_state['monitoring_log']:
        df_log = pd.DataFrame(st.session_state['monitoring_log'])
        st.download_button("📥 Download Monitoring Log", df_log.to_csv(index=False), "monitoring_log.csv")

    # System Information
    uptime = time.time() - st.session_state['start_time']
    st.markdown(f"**System Uptime:** {int(uptime//3600)}h {int((uptime%3600)//60)}m {int(uptime%60)}s")

# --- Handle Monitoring Toggle ---
if start_monitoring:
    st.session_state['monitoring'] = True
if stop_monitoring:
    st.session_state['monitoring'] = False

# --- Load Models if Not Already Loaded ---
if "Optical Camera" in sensor_selection and st.session_state['optical_model'] is None:
    with st.spinner("Loading optical model..."):
        st.session_state['optical_model'] = load_model_safe(optical_model_path)
        
if "Infrared Camera" in sensor_selection and st.session_state['ir_model'] is None:
    with st.spinner("Loading infrared model..."):
        st.session_state['ir_model'] = load_model_safe(ir_model_path)

if "Gas Detector" in sensor_selection and st.session_state['gas_model'] is None:
    with st.spinner("Loading gas model..."):
        st.session_state['gas_model'] = load_gas_model(gas_model_path)
        if st.session_state['gas_model'] is None:
            st.error(f"Failed to load gas model from {gas_model_path}")
        
        # Load the gas data
        st.session_state['gas_df'] = load_csv_with_timestamp(gas_path)
        
        # Initialize gas_index
        st.session_state['gas_index'] = 0

# --- Create Tabbed Interface for Sensor Outputs ---
tabs = st.tabs(["📷 Optical Detection", "🔥 Infrared Detection", "☁️ Gas Analysis"])

# Tab 1: Optical Camera
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="subheader">Optical Crack Detection</h3>', unsafe_allow_html=True)
    
    # Create columns for image and data
    col1, col2 = st.columns([2, 1])
    
    with col1:
        optical_section = st.empty()  # For the image display
    
    with col2:
        st.markdown("#### Crack Statistics")
        # Placeholder for crack statistics
        if st.session_state['crack_areas']:
            st.metric("Average Crack Area", f"{np.mean(st.session_state['crack_areas']):.2f} px²")
            st.metric("Average Crack Length", f"{np.mean(st.session_state['crack_lengths']):.2f} px")
            st.metric("Average Crack Width", f"{np.mean(st.session_state['crack_widths']):.2f} px")
            
            # Add a small chart
            chart_data = pd.DataFrame({
                'Area': st.session_state['crack_areas'][-10:] if len(st.session_state['crack_areas']) > 0 else [],
                'Length': st.session_state['crack_lengths'][-10:] if len(st.session_state['crack_lengths']) > 0 else [],
            })
            if not chart_data.empty:
                st.line_chart(chart_data)
        else:
            st.info("No crack data available yet")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Infrared Camera
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="subheader">Infrared Crack Detection</h3>', unsafe_allow_html=True)
    ir_section = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Gas Detector
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="subheader">Gas Concentration Analysis</h3>', unsafe_allow_html=True)
    gas_section = st.empty()
    
    # Add some visualizations for gas data
    if not st.session_state['gas_data_history'].empty:
        st.subheader("Gas Concentration Trend")
        st.line_chart(st.session_state['gas_data_history'].set_index('timestamp')['concentration'] 
                      if 'concentration' in st.session_state['gas_data_history'].columns else 
                      st.session_state['gas_data_history'])
        
        # Recent alerts
        if st.session_state['gas_alerts']:
            st.subheader("Recent Alerts")
            for alert in st.session_state['gas_alerts'][-5:]:
                st.warning(alert)
    else:
        st.info("No gas data available yet")
    st.markdown('</div>', unsafe_allow_html=True)

# --- System Status Card ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h3 class="subheader">System Status</h3>', unsafe_allow_html=True)

# Display active sensors and system info in columns
status_cols = st.columns(4)

with status_cols[0]:
    st.write("**Active Sensors:**")
    for sensor in sensor_selection:
        icon = "📷" if sensor == "Optical Camera" else "🔥" if sensor == "Infrared Camera" else "☁️"
        st.markdown(f'<span class="sensor-icon">{icon}</span> {sensor}', unsafe_allow_html=True)

with status_cols[1]:
    st.write("**Processing Information:**")
    st.write(f"Optical Files: {len(st.session_state['processed_optical_files'])}")
    st.write(f"IR Files: {len(st.session_state['processed_ir_files'])}")
    
with status_cols[2]:
    st.write("**Detection Settings:**")
    st.write(f"Confidence: {st.session_state['confidence_threshold']}%")
    if "Optical Camera" in sensor_selection:
        st.write(f"Min Area: {st.session_state['min_crack_area']} px²")
        
with status_cols[3]:
    st.write("**System Information:**")
    st.write(f"Last Update: {datetime.datetime.now().strftime('%H:%M:%S')}")
    status = "Active" if st.session_state['monitoring'] else "Inactive"
    status_class = "status-active" if st.session_state['monitoring'] else "status-inactive"
    st.markdown(f'Status: <span class="{status_class}">{status}</span>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- Monitoring Execution ---
status_message = st.empty()
if st.session_state['monitoring']:
    with st.spinner("Processing sensor data..."):
        while st.session_state['monitoring']:
            if "Optical Camera" in sensor_selection:
                process_optical_files(optical_path, optical_section)
                update_optical_section(optical_section)

            if "Infrared Camera" in sensor_selection:
                process_ir_files(ir_path, ir_section)
                update_ir_section(ir_section)

            if "Gas Detector" in sensor_selection:
                run_gas_simulator(st.session_state['gas_df'], st.session_state['gas_model'], gas_section)

            # Sleep for 2 seconds before processing the next batch
            time.sleep(2)

# --- Improved Footer ---
st.markdown("""
<div style="text-align: center; margin-top: 1rem; padding: 1rem; background-color: #f8f9fa; border-radius: 5px;">
    <p style="color: #7f8c8d; font-size: 0.8rem;">
        NDT-Rover Monitoring System © 2025 | Real-time crack detection using multi-sensor fusion technology<br>
        For more information, please refer to the documentation.
    </p>
</div>
""", unsafe_allow_html=True)
