import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BotDetectionModel:
    """
    Class to handle model loading and prediction for bot detection
    """
    def __init__(self):
        self.models = {}
        self.model_paths = {
            'DT': '../model/DT_model.pkl',
            'ET': '../model/ET_model.pkl',
            'SVM': '../model/SVM_model.pkl',
            'LR': '../model/LR_model.pkl',  # Asumsi model logistic regression juga ada
        }
        self.load_models()
        
    def load_models(self):
        """Load all available models"""
        for name, path in self.model_paths.items():
            try:
                self.models[name] = joblib.load(path)
                print(f"Successfully loaded {name} model from {path}")
            except Exception as e:
                print(f"Could not load {name} model: {e}")
    
    def preprocess_data(self, data):
        """
        Preprocess input data before prediction
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Preprocessed data ready for model prediction
        """
        # Convert IS_GLOGIN to object/string type for one-hot encoding
        data['IS_GLOGIN'] = data['IS_GLOGIN'].astype(str)
        
        # Apply one-hot encoding
        data_encoded = pd.get_dummies(data)
        
        # Check if both gender columns exist, if not add missing columns
        required_gender_cols = ['GENDER_Female', 'GENDER_Male']
        for col in required_gender_cols:
            if col not in data_encoded.columns:
                data_encoded[col] = 0
        
        # Check if both IS_GLOGIN columns exist
        required_login_cols = ['IS_GLOGIN_False', 'IS_GLOGIN_True']
        for col in required_login_cols:
            if col not in data_encoded.columns:
                data_encoded[col] = 0
        
        # Apply min-max scaling
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data_encoded)
        
        # Convert back to DataFrame with column names
        data_normalized_df = pd.DataFrame(data_normalized, columns=data_encoded.columns)
        
        return data_normalized_df
    
    def predict(self, data):
        """
        Make a prediction using all loaded models
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            dict: Prediction results with probabilities
        """
        if not self.models:
            raise ValueError("No models loaded. Cannot make predictions.")
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Get predictions from all loaded models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            try:
                # Get binary prediction
                predictions[name] = int(model.predict(processed_data)[0])
                
                # Get probability if available
                if hasattr(model, 'predict_proba'):
                    probabilities[name] = float(model.predict_proba(processed_data)[0][1])
            except Exception as e:
                print(f"Error predicting with {name} model: {e}")
        
        # Calculate ensemble prediction using majority voting
        if predictions:
            final_prediction = 1 if sum(predictions.values()) / len(predictions) >= 0.5 else 0
        else:
            raise ValueError("Could not generate predictions from any model")
        
        # Calculate average probability
        avg_probability = sum(probabilities.values()) / len(probabilities) if probabilities else None
        
        # Format result
        result = {
            "is_bot": bool(final_prediction),
            "prediction": final_prediction,
            "prediction_text": "Bot" if final_prediction == 1 else "Not Bot",
            "model_predictions": predictions,
            "model_probabilities": probabilities,
            "avg_probability": avg_probability
        }
        
        return result