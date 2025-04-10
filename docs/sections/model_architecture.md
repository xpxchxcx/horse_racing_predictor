# Model Architecture and Training

## Neural Network Architecture

### Input Layer
- 5 input features (normalized)
- Batch normalization for stable training

### Hidden Layers
1. **Dense Layer 1**
   - 128 neurons
   - ReLU activation
   - Dropout (0.3)

2. **Dense Layer 2**
   - 64 neurons
   - ReLU activation
   - Dropout (0.2)

3. **Dense Layer 3**
   - 32 neurons
   - ReLU activation
   - Dropout (0.1)

### Output Layer
- 1 neuron
- Sigmoid activation for binary classification

## Training Configuration

### Hyperparameters
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Epochs**: 100 (with early stopping)

### Regularization
- Dropout layers for preventing overfitting
- L2 regularization (lambda=0.01)
- Early stopping (patience=10)
- Learning rate reduction on plateau

## Feature Selection

### Process
1. Initial feature importance analysis using Random Forest
2. Correlation analysis to remove redundant features
3. Recursive Feature Elimination (RFE)
4. Final validation using cross-validation

### Selected Features
1. Height
2. Weight
3. Stride Length
4. Training Speed
5. Recovery Time

## Model Saving and Loading

### Save Format
- Feature selector: joblib format
- Neural network: HDF5 format (.h5)

### File Structure
```
models/
├── feature_selector.joblib
└── nn_model.h5
```

## Training Process

### Data Split
- Training: 70%
- Validation: 15%
- Test: 15%

### Cross-Validation
- 5-fold cross-validation
- Stratified splits to maintain class distribution

### Monitoring
- Training loss
- Validation loss
- Accuracy
- AUC-ROC
- Learning rate adjustments

## Model Optimization

### Techniques Applied
1. **Learning Rate Scheduling**
   - Initial rate: 0.001
   - Reduction factor: 0.1
   - Patience: 5 epochs

2. **Batch Normalization**
   - Applied after input layer
   - Improves training stability

3. **Dropout Strategy**
   - Decreasing rates through layers
   - Prevents co-adaptation

4. **Weight Initialization**
   - He initialization for ReLU layers
   - Glorot/Xavier for other layers 