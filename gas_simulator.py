import pandas as pd
import numpy as np
import time
import datetime
import streamlit as st
import plotly.graph_objects as go

# Define thresholds and labels for each gas sensor
GAS_THRESHOLDS = {
    "Gas1 PPM": {"threshold": 400, "gas": "Ammonia (NH₃)"},
    "Gas2 PPM": {"threshold": 500, "gas": "Carbon Dioxide (CO₂)"},
    "Gas3 PPM": {"threshold": 300, "gas": "Benzene (C₆H₆)"},
    "Gas4 PPM": {"threshold": 400, "gas": "Natural Gas (CH₄)"},
    "Gas5 PPM": {"threshold": 50,  "gas": "Carbon Monoxide (CO)"},
    "Gas6 PPM": {"threshold": 500, "gas": "Liquefied Petroleum Gas (LPG)"}
}



def run_gas_simulator(df, model, container):
    # Initialize session state variables if they don't exist
    if 'gas_data_history' not in st.session_state:
        st.session_state['gas_data_history'] = pd.DataFrame(columns=['Timestamp'] + list(GAS_THRESHOLDS.keys()) + ['Predicted Gas', 'Confidence (%)'])
    if 'gas_index' not in st.session_state:
        st.session_state['gas_index'] = 0
    if 'gas_alerts' not in st.session_state:
        st.session_state['gas_alerts'] = []
    
    # Check if we have data to process
    if df is None or len(df) == 0:
        st.warning("No gas sensor data available.")
        return
    
    if st.session_state['gas_index'] >= len(df):
        print(f"Debug: Gas data index ({st.session_state['gas_index']}) exceeds available data length ({len(df)})")
        st.session_state['gas_index'] = 0  # Reset index if it exceeds data length
        return


    # Get current row of data
    row = df.iloc[st.session_state['gas_index']]
    timestamp = row.get('Timestamp', datetime.datetime.now().strftime("%H:%M:%S"))
    
    # Extract gas PPM values
    # Ensure that the dataframe has the correct number of columns (6 in this case)
    gas_ppms = {}
    for key in GAS_THRESHOLDS.keys():
        if key in row:
            gas_ppms[key] = row[key]
        else:
            print(f"Debug: Missing gas column '{key}' in data")

    # Only proceed if we have all required gas data
    # Update this section of the code where gas readings are being mapped
    if len(gas_ppms) == len(GAS_THRESHOLDS):
        gas_ppm_values = np.array([list(gas_ppms.values())])
        try:
            # Ensure model is loaded
            if model is None:
                st.error("Gas detection model not loaded.")
                st.session_state['gas_index'] += 1
                return

            # Make predictions
            predicted_class = model.predict(gas_ppm_values)[0]
            print(f"Predicted class: {predicted_class}")

            predicted_proba = model.predict_proba(gas_ppm_values)[0]
            print(f"Prediction probabilities: {predicted_proba}")
            print(f"Model class labels: {model.classes_}")

            # Ensure that predicted_class is within valid bounds (0-6)
            if predicted_class >= len(model.classes_):
                st.error(f"Invalid prediction class: {predicted_class}. Max expected: {len(model.classes_)-1}")
                predicted_class = len(model.classes_) - 1  # Clamp to the last valid class

            # Handle class 6 predictions, map to Gas6 PPM
            if predicted_class == 6:
                predicted_class = 5  # Map class 6 to Gas6 PPM

            # Get predicted gas info
            predicted_gas_col = list(GAS_THRESHOLDS.keys())[predicted_class]
            predicted_gas_name = GAS_THRESHOLDS[predicted_gas_col]["gas"]
            confidence = round(predicted_proba[predicted_class] * 100, 2) if predicted_proba is not None else 100.0

            # Add new data to history
            new_data = {"Timestamp": timestamp, **gas_ppms, "Predicted Gas": predicted_gas_name, "Confidence (%)": confidence}
            st.session_state['gas_data_history'] = pd.concat(
                [st.session_state['gas_data_history'], pd.DataFrame([new_data])], 
                ignore_index=True
            )

            # Update Alerts
            current_alerts = []
            for gas_col, ppm_value in gas_ppms.items():
                if gas_col in GAS_THRESHOLDS and ppm_value >= GAS_THRESHOLDS[gas_col]['threshold']:
                    current_alerts.append(f"⚠️ {GAS_THRESHOLDS[gas_col]['gas']} exceeded threshold: {ppm_value:.1f} PPM (limit: {GAS_THRESHOLDS[gas_col]['threshold']} PPM)")
            st.session_state['gas_alerts'] = current_alerts

        except Exception as e:
            import traceback
            st.error(f"Error during gas prediction: {e}")
            print(f"Debug: Gas prediction error - {traceback.format_exc()}")
    else:
        st.warning(f"Incomplete gas data. Expected {len(GAS_THRESHOLDS)} gas readings, found {len(gas_ppms)}.")




    # Move to next row for next update
    st.session_state['gas_index'] += 1

    # Update the UI
    # Update the UI
    with container.container():
        st.markdown("### 💨 Gas Sensor Readings")
        if not st.session_state['gas_data_history'].empty:
            # Update Chart
            fig = go.Figure()
            for idx, (gas_col, info) in enumerate(GAS_THRESHOLDS.items()):
                if gas_col in st.session_state['gas_data_history'].columns:
                    fig.add_trace(go.Scatter(
                        x=st.session_state['gas_data_history']["Timestamp"], 
                        y=st.session_state['gas_data_history'][gas_col], 
                        mode='lines+markers', 
                        name=f"{info['gas']}",
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
                    
                    # Add threshold line
                    fig.add_shape(
                        type="line",
                        x0=st.session_state['gas_data_history']["Timestamp"].iloc[0],
                        y0=info['threshold'],
                        x1=st.session_state['gas_data_history']["Timestamp"].iloc[-1],
                        y1=info['threshold'],
                        line=dict(color="red", width=1, dash="dash"),
                        name=f"{info['gas']} Threshold"
                    )
                    
            fig.update_layout(
                title="Real-time Gas Concentrations",
                xaxis_title="Time",
                yaxis_title="PPM",
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Provide a unique key for each chart element by using both `idx` and the current timestamp
            unique_key = f"gas_sensor_plot_{idx}_{datetime.datetime.now().timestamp()}"
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
            
            # Update Table with proper styling
            st.dataframe(
                st.session_state['gas_data_history'].tail(10).style
                .background_gradient(subset=[col for col in GAS_THRESHOLDS.keys() if col in st.session_state['gas_data_history'].columns], cmap='YlOrRd')
                .format({col: "{:.1f}" for col in GAS_THRESHOLDS.keys() if col in st.session_state['gas_data_history'].columns})
                .format({"Confidence (%)": "{:.1f}%"}),
                use_container_width=True
            )
        else:
            st.info("No gas data available yet.")




        # Display Alerts
        if st.session_state['gas_alerts']:
            st.error(" \n".join(st.session_state['gas_alerts']))
        else:
            st.success("No gas alerts.")
