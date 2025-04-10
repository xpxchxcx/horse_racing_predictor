import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import io
from kaggle.api.kaggle_api_extended import KaggleApi

def download_horse_data():
    """
    Downloads Hong Kong horse racing dataset from Kaggle.
    Dataset: https://www.kaggle.com/datasets/gdaley/hkracing
    Contains races.csv and runs.csv
    """
    try:
        print("Authenticating with Kaggle...")
        api = KaggleApi()
        api.authenticate()
        
        # Create data directory if it doesn't exist
        data_path = Path("data")
        data_path.mkdir(exist_ok=True)
        
        print("Downloading Hong Kong horse racing dataset...")
        api.dataset_download_files(
            'gdaley/hkracing',
            path='data',
            unzip=True
        )
        
        required_files = ['races.csv', 'runs.csv']
        missing_files = [f for f in required_files if not (data_path / f).exists()]
        
        if missing_files:
            raise Exception(f"Missing required files: {missing_files}")
        else:
            print("Data downloaded successfully!")
            return True
            
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Using backup sample data instead...")
        return False

def safe_qcut(series, n, labels):
    """Safely create quantile categories handling duplicate values"""
    try:
        return pd.qcut(series, n, labels=labels)
    except ValueError:
        # If we get duplicate values, use rank to break ties
        ranked = series.rank(method='first')
        return pd.qcut(ranked, n, labels=labels)

def process_real_data(races_df, runs_df):
    """
    Process the Hong Kong racing dataset into our required format.
    
    Parameters:
    - races_df: DataFrame containing race information
    - runs_df: DataFrame containing individual horse runs in races
    
    Returns:
    - Processed DataFrame with features for model training
    """
    print("Processing Hong Kong racing data...")
    
    # Merge the dataframes
    print("Merging datasets...")
    df = runs_df.merge(races_df, on='race_id', suffixes=('_run', '_race'))
    
    # Calculate win rate for each horse
    print("Calculating horse statistics...")
    horse_stats = runs_df.groupby('horse_id').agg({
        'won': ['count', 'sum']
    }).reset_index()
    horse_stats.columns = ['horse_id', 'total_races', 'wins']
    horse_stats['win_rate'] = horse_stats['wins'] / horse_stats['total_races']
    
    # Calculate average speed (distance/finish_time)
    df['speed'] = df['distance'] / df['finish_time']
    
    # Merge win rates
    df = df.merge(horse_stats, on='horse_id', suffixes=('', '_stats'))
    
    # Extract and transform features
    print("Extracting features...")
    
    # Convert declared_weight from pounds to a reasonable height estimate in cm
    # Assuming correlation between weight and height, using typical Thoroughbred proportions
    avg_height = 163  # Average Thoroughbred height in cm
    avg_weight = 1000  # Average declared weight in pounds
    height_scale = avg_height / avg_weight
    
    processed_data = {
        'height': df['declared_weight'].fillna(df['declared_weight'].mean()) * height_scale,  # Convert to reasonable height in cm
        'weight': df['actual_weight'].fillna(df['actual_weight'].mean()) * 0.45359237,  # Convert pounds to kg
        'stride_length': df['speed'] / df['speed'].mean(),  # Normalized speed as proxy for stride length
        'training_speed': df['speed'].fillna(df['speed'].mean()),
        'recovery_time': 20 - (df['win_rate'] * 10),  # Estimated from win rate
        'heart_rate': 130 + (df['speed'] / df['speed'].max() * 40),  # Estimated from speed
        'age_months': df['horse_age'] * 12,  # Convert age to months
        'breed': 'Thoroughbred',  # Hong Kong races are Thoroughbred only
        'temperament': safe_qcut(df['speed'], 3, ['Calm', 'Energetic', 'Nervous']),
        'training_response': safe_qcut(df['horse_rating'], 4, ['Poor', 'Fair', 'Good', 'Excellent']),
        'parent1_performance': 'Unknown',  # Not available in dataset
        'parent2_performance': 'Unknown',  # Not available in dataset
        'success': df['won'].astype(int)  # 1 if won, 0 otherwise
    }
    
    result_df = pd.DataFrame(processed_data)
    
    # Print data statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(result_df)}")
    print(f"Unique horses: {df['horse_id'].nunique()}")
    print(f"Total races: {df['race_id'].nunique()}")
    print(f"Positive samples (wins): {result_df['success'].sum()}")
    print(f"Win rate: {result_df['success'].mean():.2%}")
    print(f"Average height (cm): {result_df['height'].mean():.1f}")
    print(f"Average weight (kg): {result_df['weight'].mean():.1f}")
    print(f"Average training speed: {result_df['training_speed'].mean():.1f}")
    
    return result_df

def load_real_data():
    """
    Loads and preprocesses Hong Kong horse racing data.
    Falls back to sample data if real data is not available.
    """
    try:
        print("Loading Hong Kong racing data...")
        races_df = pd.read_csv('data/races.csv')
        runs_df = pd.read_csv('data/runs.csv')
        
        print(f"Loaded {len(races_df)} races and {len(runs_df)} runs")
        
        # Process the data into our required format
        processed_df = process_real_data(races_df, runs_df)
        
        print(f"Successfully processed {len(processed_df)} race records!")
        return processed_df
            
    except Exception as e:
        print(f"Error loading Hong Kong racing data: {e}")
        print("Falling back to sample data...")
        return load_sample_data()

def load_sample_data():
    """
    Creates sample data for demonstration when real data is not available.
    Mimics the structure of the Hong Kong racing dataset.
    """
    print("Generating sample data...")
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'height': np.random.normal(160, 10, n_samples),  # cm
        'weight': np.random.normal(500, 50, n_samples),  # kg
        'stride_length': np.random.normal(2.5, 0.3, n_samples),  # meters
        'training_speed': np.random.normal(40, 5, n_samples),  # km/h
        'recovery_time': np.random.normal(15, 3, n_samples),  # minutes
        'heart_rate': np.random.normal(140, 20, n_samples),  # bpm
        'age_months': np.random.randint(24, 48, n_samples),
        'breed': 'Thoroughbred',  # All horses in HK racing are Thoroughbred
        'temperament': np.random.choice(['Calm', 'Energetic', 'Nervous'], n_samples),
        'training_response': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples),
        'parent1_performance': 'Unknown',  # Matching real data
        'parent2_performance': 'Unknown',  # Matching real data
    }
    
    # Generate target variable (success in racing)
    success_probability = (
        0.3 * ((data['height'] - 160) / 10) +
        0.2 * ((data['weight'] - 500) / 50) +
        0.2 * ((data['stride_length'] - 2.5) / 0.3) +
        0.3 * ((data['training_speed'] - 40) / 5)
    )
    
    data['success'] = (success_probability + np.random.normal(0, 0.5, n_samples) > 0).astype(int)
    
    print(f"Generated {n_samples} sample records")
    return pd.DataFrame(data) 