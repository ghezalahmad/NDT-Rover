# Re-import and define the GasDetectionModel class for retraining the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report  # Add this import


class GasDetectionModel:
    def __init__(self, model_path='gas_model.pkl', scaler_path='gas_scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.gas_columns = ['Gas1 PPM', 'Gas2 PPM', 'Gas3 PPM', 'Gas4 PPM', 'Gas5 PPM', 'Gas6 PPM']
    
    def load_gas_data(self, file_path):
        """Load and prepare gas sensor dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"Debug: Loaded gas data with shape {df.shape}")
            
            # Check if required columns exist
            missing_cols = [col for col in self.gas_columns + ['Class'] if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in gas data: {missing_cols}")
            
            # Select relevant columns: Gas PPM columns and 'Class' for training
            X = df[self.gas_columns]
            y = df['Class']
            
            # Create and fit the scaler
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            print(f"Debug: Processed X data shape: {X_scaled.shape}, y shape: {y.shape}")
            return X_scaled, y
            
        except Exception as e:
            print(f"Error loading gas data: {e}")
            return None, None

    def train_model(self, file_path='gas_data/MQ135SensorData.csv', save_model=True):
        """Train a new gas detection model"""
        X, y = self.load_gas_data(file_path)
        
        if X is None or y is None:
            print("Failed to load training data.")
            return False
            
        try:
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train a RandomForest Classifier
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model accuracy: {accuracy:.4f}")
            print(classification_report(y_test, y_pred))
            
            # Save the trained model if requested
            if save_model:
                self.save_model()
                
            return True
            
        except Exception as e:
            print(f"Error training gas model: {e}")
            return False
    
    def save_model(self):
        """Save the model and scaler to disk"""
        try:
            if self.model is not None:
                joblib.dump(self.model, self.model_path)
                print(f"Model saved to {self.model_path}")
            
            if self.scaler is not None:
                joblib.dump(self.scaler, self.scaler_path)
                print(f"Scaler saved to {self.scaler_path}")
                
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load the trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"Model loaded from {self.model_path}")
            else:
                print(f"Model file not found: {self.model_path}")
                return False
                
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print(f"Scaler loaded from {self.scaler_path}")
            else:
                print(f"Scaler file not found, preprocessing may be inaccurate")
                
            return self.model is not None
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_data(self, sensor_data):
        """Preprocess raw sensor data"""
        if self.scaler is not None:
            # If input is a dictionary, convert to array
            if isinstance(sensor_data, dict):
                # Ensure data is in the correct order
                ordered_data = [sensor_data.get(col, 0) for col in self.gas_columns]
                sensor_data = np.array([ordered_data])
                
            return self.scaler.transform(sensor_data)
        else:
            print("Warning: Scaler not loaded, using raw values")
            return sensor_data
    
    def predict(self, sensor_data):
        """Predict gas class from sensor data"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Preprocess the data if scaler exists
        processed_data = self.preprocess_data(sensor_data) if self.scaler else sensor_data
        
        # Make prediction
        return self.model.predict(processed_data)
    
    def predict_proba(self, sensor_data):
        """Get probability estimates for each class"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Preprocess the data if scaler exists
        processed_data = self.preprocess_data(sensor_data) if self.scaler else sensor_data
        
        # Make probability prediction
        return self.model.predict_proba(processed_data)

if __name__ == '__main__':
    # Define the file paths for saving the model
    model_path = 'gas_model.pkl'
    scaler_path = 'gas_scaler.pkl'

    # Create an instance of the model
    gas_model = GasDetectionModel(model_path=model_path, scaler_path=scaler_path)
    
    # Train the model (this will save the model and scaler if successful)
    success = gas_model.train_model(file_path='gas_data/MQ135SensorData.csv')

    if success:
        print("Model training completed and saved successfully!")
    else:
        print("Model training failed.")
