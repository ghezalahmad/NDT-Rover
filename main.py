import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="Digital Twin NDT Simulation", layout="wide")
st.title("Climbing Robot Digital Twin - Realistic NDT Simulation with ML")

st.markdown("""
Physics-inspired simulation of NDT sensors mounted on an autonomous climbing robot.
Sensors include: Infrared Camera, Ultrasonic Sensor, Optical Camera, Humidity Sensor, and Gas Detector.
Users can view individual sensor outputs or combined system performance.
Image analysis is applied to detect and classify cracks in real time.
""")

sensor_tooltips = {
    "All Sensors": "View all sensors together on the same chart.",
    "Infrared Camera (Temperature)": "Simulates surface temperature gradients. IR cameras normally produce thermal images. Random image shown as proxy.",
    "Ultrasonic Sensor (Thickness)": "Estimates material thickness from time-of-flight echo delay using wave velocity.",
    "Optical Camera (Crack Intensity)": "Detects surface cracks using image processing and visual pattern analysis.",
    "Humidity Sensor": "Measures relative humidity, affects corrosion and material behavior.",
    "Gas Detector": "Detects gas emissions indicating chemical degradation or leaks.",
    "Corrosion Index": "Synthetic metric representing likelihood of corrosion based on environmental parameters."
}

sensor_options = list(sensor_tooltips.keys())
selected_sensor = st.sidebar.selectbox("Select Sensor View", sensor_options)
st.sidebar.caption(sensor_tooltips[selected_sensor])

sample_images = [
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/320px-Golde33443.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Cracked_Concrete.jpg/320px-Cracked_Concrete.jpg"
]

def simulate_ultrasonic_thickness():
    base_thickness = 5.0
    echo_delay = np.random.normal(1.2, 0.1)
    velocity = 5900
    thickness = (echo_delay * 1e-6 * velocity) / 2 * 1000
    return round(base_thickness - np.clip(base_thickness - thickness, 0, base_thickness), 2)

def dummy_crack_detection(image_url):
    if "Cracked" in image_url:
        return 1.0  # simulate crack detection
    return 0.1  # simulate no crack detected

def generate_sensor_data():
    image_url = random.choice(sample_images)
    crack_intensity = dummy_crack_detection(image_url)
    gas_level = np.random.normal(9, 3)
    corrosion_index = np.clip(np.random.beta(2, 8), 0, 1)
    thickness = simulate_ultrasonic_thickness()

    fault_type = "None"
    crack_size = 0
    severity_score = 0

    if crack_intensity > 0.35:
        fault_type = "Crack"
        crack_size = round(crack_intensity * 12, 2)
        severity_score = crack_intensity * 100
    elif corrosion_index > 0.25:
        fault_type = "Corrosion"
        severity_score = corrosion_index * 200
    elif gas_level > 14:
        fault_type = "Gas Leak"
        severity_score = gas_level * 10

    return {
        "temperature (°C)": np.random.normal(25, 1.5),
        "thickness (mm)": thickness,
        "crack intensity": crack_intensity,
        "humidity (%)": np.random.normal(50, 8),
        "gas emission (ppm)": gas_level,
        "corrosion index": corrosion_index,
        "crack size (mm)": crack_size,
        "fault type": fault_type,
        "severity score": round(severity_score, 1),
        "image url": image_url
    }

def assign_priority(severity):
    if severity > 40:
        return "Immediate"
    elif severity > 20:
        return "Urgent"
    elif severity > 5:
        return "Scheduled"
    else:
        return "Monitor"

st.sidebar.header("Simulation Control")
running = st.sidebar.button("Start Real-Time Simulation")

model = RandomForestClassifier(n_estimators=50)
scaler = StandardScaler()

def predict_risk_ml(model, scaler, sample):
    features = sample[feature_cols].values.reshape(1, -1)
    scaled = scaler.transform(features)
    return model.predict(scaled)[0]

feature_cols = [
    "temperature (°C)", "thickness (mm)", "crack intensity",
    "humidity (%)", "gas emission (ppm)", "corrosion index"
]

weights = {col: 1.0 for col in feature_cols}

data_log = []
plot_placeholder = st.empty()
map_placeholder = st.empty()
image_placeholder = st.empty()

path = [(i, j) for i in range(10) for j in range(5)]
zones = {(i, j): f"Zone-{i}-{j}" for i, j in path}

if running:
    for i in range(50):
        new_data = generate_sensor_data()
        x, y = path[i % len(path)]
        new_data.update({
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
            "x": x, "y": y,
            "zone": zones[(x, y)],
            "priority": assign_priority(new_data["severity score"])
        })

        if new_data["crack intensity"] > 0.6 or new_data["gas emission (ppm)"] > 20 or new_data["corrosion index"] > 0.4:
            new_data["risk"] = "High"
        elif new_data["crack intensity"] > 0.35 or new_data["gas emission (ppm)"] > 14 or new_data["corrosion index"] > 0.25:
            new_data["risk"] = "Moderate"
        else:
            new_data["risk"] = "Low"

        data_log.append(new_data)
        df = pd.DataFrame(data_log)

        if len(df) >= 10:
            X = df[feature_cols]
            y = df["risk"]
            scaler.fit(X)
            model.fit(scaler.transform(X), y)
            df["risk"] = df.apply(lambda row: predict_risk_ml(model, scaler, row), axis=1)

        fig = go.Figure()
        selected_cols = feature_cols if selected_sensor == "All Sensors" else [col for col in feature_cols if selected_sensor.lower() in col.lower()]

        for col in selected_cols:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col], mode='lines+markers', name=col))

        color_map = {"Low": "green", "Moderate": "orange", "High": "red"}
        max_vals = [df[col].max() for col in selected_cols if col in df]
        y_risk_level = [1.05 * max(max_vals)] * len(df) if max_vals else [0] * len(df)

        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=y_risk_level,
            mode='markers',
            marker=dict(size=10, color=[color_map[r] for r in df["risk"]]),
            name="Risk Level"
        ))

        fig.update_layout(title=f"{selected_sensor} - Sensor Trend with Risk", height=600)
        plot_placeholder.plotly_chart(fig, use_container_width=True)

        map_fig = px.scatter(df, x="x", y="y", color="risk",
                             hover_data=["zone", "fault type", "crack size (mm)", "severity score", "priority", "timestamp"],
                             color_discrete_map=color_map,
                             title="Spatial Risk Map with Fault Annotations",
                             width=700, height=500)
        map_placeholder.plotly_chart(map_fig, use_container_width=True)

        if selected_sensor.lower().startswith("infrared") or selected_sensor.lower().startswith("optical"):
            try:
                response = requests.get(new_data["image url"])
                image = Image.open(BytesIO(response.content))
                image_placeholder.image(image, caption=f"Simulated frame for {selected_sensor}", width=250)
            except:
                image_placeholder.error("Unable to load image.")

        time.sleep(0.3)

    st.success("Simulation complete.")
    st.dataframe(df.tail(10))
else:
    st.info("Click 'Start Real-Time Simulation' to begin.")
