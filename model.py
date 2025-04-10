import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

class HorseRacingPredictor:
    def __init__(self, feature_selector_path=None, nn_model_path=None):
        # Number of features in our dataset
        self.n_features = 12  # Total number of features we're using
        
        self.feature_selector = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.nn_model = self._build_neural_network()
        
        if feature_selector_path:
            self.feature_selector = joblib.load(feature_selector_path)
        if nn_model_path:
            self.nn_model.load_weights(nn_model_path)
            
    def _build_neural_network(self):
        model = Sequential([
            # Input layer with specific shape (number of features)
            Dense(128, activation='relu', input_shape=(self.n_features,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def _plot_feature_importance(self, feature_importance, feature_names, save_path='feature_importance.png'):
        """Plot feature importance"""
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        sns.barplot(data=importance_df, y='feature', x='importance')
        plt.title('Feature Importance Analysis')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def _plot_roc_curve(self, y_true, y_pred_proba, save_path='roc_curve.png'):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()
    
    def _plot_precision_recall(self, y_true, y_pred_proba, save_path='precision_recall.png'):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(save_path)
        plt.close()
    
    def _perform_cross_validation(self, X, y, n_splits=5):
        """Perform cross-validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train feature selector
            self.feature_selector.fit(X_train_fold, y_train_fold)
            important_features = self.feature_selector.feature_importances_ > np.mean(self.feature_selector.feature_importances_)
            
            # Select features
            X_train_selected = X_train_fold[:, important_features]
            X_val_selected = X_val_fold[:, important_features]
            
            # Train neural network
            self.nn_model = Sequential([
                Dense(128, activation='relu', input_shape=(np.sum(important_features),)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            self.nn_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'AUC']
            )
            
            history = self.nn_model.fit(
                X_train_selected, y_train_fold,
                validation_data=(X_val_selected, y_val_fold),
                epochs=50,
                batch_size=32,
                verbose=0
            )
            
            # Get validation score
            val_score = self.nn_model.evaluate(X_val_selected, y_val_fold, verbose=0)[1]
            cv_scores.append(val_score)
            
            print(f"Fold {fold + 1}/5: Validation Accuracy = {val_score:.4f}")
        
        return cv_scores
    
    def train_and_evaluate(self, X, y, test_size=0.2, validation_split=0.2, feature_names=None):
        """Train the model and evaluate on test set with comprehensive analysis"""
        # Perform cross-validation first
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = self._perform_cross_validation(X, y)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=validation_split,
            random_state=42
        )
        
        # Train feature selector
        self.feature_selector.fit(X_train, y_train)
        feature_importance = self.feature_selector.feature_importances_
        
        # Plot feature importance if feature names are provided
        if feature_names is not None:
            self._plot_feature_importance(feature_importance, feature_names)
        
        # Select important features
        important_features = feature_importance > np.mean(feature_importance)
        X_train_selected = X_train[:, important_features]
        X_val_selected = X_val[:, important_features]
        X_test_selected = X_test[:, important_features]
        
        # Train neural network
        n_selected_features = np.sum(important_features)
        self.nn_model = Sequential([
            Dense(128, activation='relu', input_shape=(n_selected_features,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.nn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        history = self.nn_model.fit(
            X_train_selected, y_train,
            validation_data=(X_val_selected, y_val),
            epochs=300,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Get predictions
        test_predictions = self.nn_model.predict(X_test_selected)
        test_predictions_binary = (test_predictions > 0.5).astype(int)
        
        # Calculate metrics
        test_metrics = {
            'accuracy': self.nn_model.evaluate(X_test_selected, y_test)[1],
            'auc_roc': roc_auc_score(y_test, test_predictions),
            'classification_report': classification_report(y_test, test_predictions_binary),
            'confusion_matrix': confusion_matrix(y_test, test_predictions_binary)
        }
        
        # Plot ROC curve
        self._plot_roc_curve(y_test, test_predictions)
        
        # Plot Precision-Recall curve
        self._plot_precision_recall(y_test, test_predictions)
        
        # Plot training history
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history.history['auc'], label='Training AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            test_metrics['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive']
        )
        plt.title('Confusion Matrix on Test Set')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Statistical analysis
        confidence_interval = stats.norm.interval(
            0.95, 
            loc=test_metrics['accuracy'], 
            scale=stats.sem([1 if p == t else 0 for p, t in zip(test_predictions_binary, y_test)])
        )
        
        print("\n=== Model Performance Analysis ===")
        print(f"\nTest Set Metrics:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"95% Confidence Interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
        print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"\nCross-validation Results:")
        print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
        print(f"CV Standard Deviation: {np.std(cv_scores):.4f}")
        print("\nClassification Report:")
        print(test_metrics['classification_report'])
        
        return history, feature_importance, test_metrics, cv_scores
    
    def predict(self, X):
        """Predict racing success probability"""
        feature_importance = self.feature_selector.feature_importances_
        important_features = feature_importance > np.mean(feature_importance)
        X_selected = X[:, important_features]
        return self.nn_model.predict(X_selected)
    
    def save_models(self, feature_selector_path, nn_model_path):
        """Save both models to disk"""
        joblib.dump(self.feature_selector, feature_selector_path)
        self.nn_model.save_weights(nn_model_path)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        feature_importance = self.feature_selector.feature_importances_
        important_features = feature_importance > np.mean(feature_importance)
        X_test_selected = X_test[:, important_features]
        
        predictions = self.nn_model.predict(X_test_selected)
        predictions_binary = (predictions > 0.5).astype(int)
        
        metrics = {
            'accuracy': self.nn_model.evaluate(X_test_selected, y_test)[1],
            'auc_roc': roc_auc_score(y_test, predictions),
            'classification_report': classification_report(y_test, predictions_binary),
            'confusion_matrix': confusion_matrix(y_test, predictions_binary)
        }
        
        return metrics 