import pandas as pd
import numpy as np
from data_preprocessing import HorseDataPreprocessor
from model import HorseRacingPredictor
from data_loader import load_real_data, download_horse_data
import joblib
from pathlib import Path

def analyze_results(test_metrics, cv_scores):
    """Generate a detailed analysis report"""
    with open('model_analysis_template.md', 'r') as f:
        template = f.read()
    
    # Format the analysis with actual values using named arguments
    analysis = template.format(
        mean_cv=np.mean(cv_scores),
        std_cv=np.std(cv_scores),
        ci_lower=np.mean(cv_scores) - 1.96 * np.std(cv_scores),
        ci_upper=np.mean(cv_scores) + 1.96 * np.std(cv_scores),
        test_acc=test_metrics['accuracy'],
        auc_roc=test_metrics['auc_roc'],
        cv_std=np.std(cv_scores),
        stability='highly ' if np.std(cv_scores) < 0.05 else 'moderately ' if np.std(cv_scores) < 0.1 else 'somewhat ',
        interpretation=get_interpretation(test_metrics),
        recommendations=get_recommendations(test_metrics)
    )
    
    # Save analysis to file
    with open('model_analysis.md', 'w') as f:
        f.write(analysis)
    
    return analysis

def get_interpretation(test_metrics):
    """Generate interpretation based on model metrics"""
    acc = test_metrics['accuracy']
    auc = test_metrics['auc_roc']
    
    if acc >= 0.8 and auc >= 0.8:
        return """The model shows strong predictive performance with both high accuracy and AUC-ROC scores.
This indicates reliable predictions across both positive and negative cases."""
    elif acc >= 0.7 and auc >= 0.7:
        return """The model shows good predictive performance with reasonable accuracy and AUC-ROC scores.
There is room for improvement but the current performance is practical for many applications."""
    else:
        return """The model shows moderate predictive performance.
Consider gathering more data or feature engineering to improve the results."""

def get_recommendations(test_metrics):
    """Generate recommendations based on model performance"""
    acc = test_metrics['accuracy']
    auc = test_metrics['auc_roc']
    
    recommendations = []
    
    if acc < 0.7:
        recommendations.append("- Consider collecting more training data to improve accuracy")
        recommendations.append("- Experiment with feature engineering to create more informative features")
    
    if auc < 0.8:
        recommendations.append("- Fine-tune the model's threshold for better precision-recall balance")
        recommendations.append("- Investigate potential class imbalance in the training data")
    
    if not recommendations:
        recommendations.append("- Model is performing well; monitor for potential drift in production")
        recommendations.append("- Consider ensemble methods to potentially improve robustness")
    
    return "\n".join(recommendations)

def main():
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download and load data
    print("Attempting to download horse racing data...")
    download_horse_data()
    
    print("Loading data...")
    data = load_real_data()
    
    # Initialize preprocessor and predictor
    print("Initializing models...")
    preprocessor = HorseDataPreprocessor()
    predictor = HorseRacingPredictor()
    
    # Prepare features and target
    print("Preprocessing data...")
    X = preprocessor.fit_transform(data.drop('success', axis=1))
    y = data['success']
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Train and evaluate the model with feature names
    print("\nTraining and evaluating model...")
    feature_names = preprocessor.get_feature_names()
    history, feature_importance, test_metrics, cv_scores = predictor.train_and_evaluate(
        X.values, y.values, feature_names=feature_names
    )
    
    # Save the models
    print("\nSaving models...")
    predictor.save_models('models/feature_selector.joblib', 'models/nn_model.h5')
    
    # Generate and save analysis
    print("\nGenerating comprehensive analysis...")
    analysis = analyze_results(test_metrics, cv_scores)
    
    # Print feature importance
    print("\nFeature Importance Analysis:")
    features_to_display = [
        'height', 'weight', 'stride_length', 'training_speed',
        'recovery_time', 'heart_rate', 'age_months', 'breed',
        'temperament', 'training_response'
    ]
    for name, importance in zip(features_to_display, feature_importance[:len(features_to_display)]):
        print(f"{name}: {importance:.4f}")
    
    # Example prediction
    print("\nExample prediction for a new horse:")
    new_horse = pd.DataFrame({
        'height': [162],
        'weight': [510],
        'stride_length': [2.6],
        'training_speed': [42],
        'recovery_time': [14],
        'heart_rate': [135],
        'age_months': [30],
        'breed': ['Thoroughbred'],
        'temperament': ['Energetic'],
        'training_response': ['Good']
    })
    
    # Preprocess and predict
    processed_data = preprocessor.transform(new_horse)
    success_probability = predictor.predict(processed_data.values)[0][0]
    
    print(f"\nSuccess Probability: {success_probability:.2%}")
    
    # Generate recommendations
    if success_probability >= 0.7:
        print("\nRecommendations:")
        print("- High potential for racing success")
        print("- Consider specialized training program")
        print("- Focus on maintaining current performance levels")
    elif success_probability >= 0.4:
        print("\nRecommendations:")
        print("- Moderate potential for racing")
        print("- Focus on improving speed and endurance")
        print("- Consider additional training in areas of weakness")
    else:
        print("\nRecommendations:")
        print("- Lower racing potential")
        print("- Consider alternative disciplines")
        print("- Focus on general fitness and well-being")
    
    print("\nAnalysis and visualization files generated:")
    print("- model_analysis.md (Comprehensive analysis report)")
    print("- training_history.png (Loss, Accuracy, and AUC curves)")
    print("- confusion_matrix.png (Test set performance)")
    print("- roc_curve.png (ROC curve analysis)")
    print("- precision_recall.png (Precision-Recall curve)")
    print("- feature_importance.png (Feature importance analysis)")

if __name__ == "__main__":
    main() 