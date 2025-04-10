import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import datetime

def load_model_safe(path, compile=False):
    try:
        return load_model(path, compile=compile)
    except Exception as e:
        st.error(f"Error loading model {path}: {e}")
        return None

def load_gas_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading gas model {path}: {e}")
        return None

@st.cache_data
def load_csv_with_timestamp(path):
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        if 'Date' in df.columns and 'Time(sec)' in df.columns:
            def create_timestamp(row):
                try:
                    date_part = row['Date']
                    time_sec = float(row['Time(sec)'])
                    try:
                        base_datetime = datetime.datetime.strptime(date_part, '%d/%m/%Y')
                    except ValueError:
                        try:
                            base_datetime = datetime.datetime.strptime(date_part, '%Y-%m-%d')
                        except ValueError:
                            return None  # Or handle other date formats

                    timestamp = base_datetime + datetime.timedelta(seconds=time_sec)
                    return timestamp.strftime('%H:%M:%S')
                except Exception as e:
                    print(f"Error creating timestamp for row: {row}. Error: {e}")
                    return None

            df['Timestamp'] = df.apply(create_timestamp, axis=1)
            df = df.dropna(subset=['Timestamp'])
        elif 'Timestamp' in df.columns:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                df['Timestamp'] = df['Timestamp'].dt.strftime('%H:%M:%S')
                df = df.dropna(subset=['Timestamp'])
            except Exception as e:
                st.error(f"Error processing 'Timestamp' column: {e}")
                return pd.DataFrame()
        else:
            st.error(f"Neither 'Timestamp' nor 'Date' and 'Time(sec)' columns found in {path}")
            return pd.DataFrame()

        # Check for the required PPM columns
        ppm_columns = ["Gas1 PPM", "Gas2 PPM", "Gas3 PPM", "Gas4 PPM", "Gas5 PPM", "Gas6 PPM"]
        missing_ppm_columns = [col for col in ppm_columns if col not in df.columns]

        if missing_ppm_columns:
            # Try to use raw gas values if PPM columns are missing
            raw_gas_columns = ["Gas1", "Gas2", "Gas3", "Gas4", "Gas5", "Gas6"]
            rename_dict = {}
            found_raw = True
            for i, ppm_col in enumerate(ppm_columns):
                if ppm_col not in df.columns and raw_gas_columns[i] in df.columns:
                    rename_dict[raw_gas_columns[i]] = ppm_col
                elif ppm_col not in df.columns and raw_gas_columns[i] not in df.columns:
                    found_raw = False
                    break
            df = df.rename(columns=rename_dict)
            missing_after_rename = [col for col in ppm_columns if col not in df.columns]
            if missing_after_rename:
                st.warning(f"Missing PPM columns: {missing_after_rename}. Gas detection might not work as expected.")
                # You might want to fill these with NaN or 0 depending on your model's behavior
                for col in missing_after_rename:
                    df[col] = 0.0

        return df

    except FileNotFoundError:
        st.error(f"Gas data file not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading gas data from {path}: {e}")
        return pd.DataFrame()