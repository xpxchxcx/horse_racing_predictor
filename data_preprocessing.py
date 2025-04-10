import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class HorseDataPreprocessor:
    def __init__(self):
        self.numerical_features = [
            'height', 'weight', 'stride_length', 'training_speed',
            'recovery_time', 'heart_rate', 'age_months'
        ]
        
        self.categorical_features = [
            'breed', 'temperament', 'training_response'
        ]
        
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def fit(self, data):
        """Fit preprocessors on training data"""
        # Initialize and fit imputers
        for feature in self.numerical_features:
            self.imputers[feature] = SimpleImputer(strategy='mean')
            self.imputers[feature].fit(data[[feature]])
            
        for feature in self.categorical_features:
            self.imputers[feature] = SimpleImputer(strategy='most_frequent')
            self.imputers[feature].fit(data[[feature]])
        
        # Initialize and fit scalers for numerical features
        for feature in self.numerical_features:
            self.scalers[feature] = StandardScaler()
            imputed_data = self.imputers[feature].transform(data[[feature]])
            self.scalers[feature].fit(imputed_data)
        
        # Initialize and fit encoders for categorical features
        for feature in self.categorical_features:
            self.encoders[feature] = LabelEncoder()
            imputed_data = self.imputers[feature].transform(data[[feature]])
            self.encoders[feature].fit(imputed_data.ravel())
    
    def transform(self, data):
        """Transform new data using fitted preprocessors"""
        processed_data = pd.DataFrame()
        
        # Process numerical features
        for feature in self.numerical_features:
            imputed_data = self.imputers[feature].transform(data[[feature]])
            scaled_data = self.scalers[feature].transform(imputed_data)
            processed_data[feature] = scaled_data.ravel()
        
        # Process categorical features
        for feature in self.categorical_features:
            imputed_data = self.imputers[feature].transform(data[[feature]])
            encoded_data = self.encoders[feature].transform(imputed_data.ravel())
            processed_data[feature] = encoded_data
        
        return processed_data
    
    def fit_transform(self, data):
        """Fit preprocessors and transform data"""
        self.fit(data)
        return self.transform(data)
    
    def get_feature_names(self):
        """Return list of all features"""
        return self.numerical_features + self.categorical_features

    def calculate_derived_features(self, data):
        """Calculate additional features from raw data"""
        derived_features = pd.DataFrame()
        
        # Speed to weight ratio (power indicator)
        if 'training_speed' in data.columns and 'weight' in data.columns:
            derived_features['speed_weight_ratio'] = data['training_speed'] / data['weight']
        
        # Stride efficiency (stride length to height ratio)
        if 'stride_length' in data.columns and 'height' in data.columns:
            derived_features['stride_efficiency'] = data['stride_length'] / data['height']
        
        # Recovery efficiency (inverse of recovery time * heart rate)
        if 'recovery_time' in data.columns and 'heart_rate' in data.columns:
            derived_features['recovery_efficiency'] = 1 / (data['recovery_time'] * data['heart_rate'])
        
        return derived_features 