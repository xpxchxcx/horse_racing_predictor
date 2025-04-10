# Data Engineering

## Feature Engineering Process

1. **Physical Characteristics**
   - **Height**: Derived from declared weight using typical Thoroughbred proportions
     - Average height: 163 cm (typical for Thoroughbreds)
     - Converted from declared weight using scaling factor
     - Mean height in dataset: 180.1 cm
   
   - **Weight**: Converted from pounds to kilograms
     - Used actual_weight from runs.csv
     - Applied standard conversion (× 0.45359237)
     - Mean weight: 55.7 kg (includes jockey and equipment)

2. **Performance Metrics**
   - **Speed Calculation**
     ```python
     speed = distance / finish_time
     ```
   
   - **Stride Length**: Normalized speed as proxy
     ```python
     stride_length = speed / speed.mean()
     ```
   
   - **Training Speed**: Direct from speed calculations
     - Filled missing values with mean
     - Used as key performance indicator

3. **Derived Features**
   - **Win Rate**: Calculated per horse
     ```python
     win_rate = wins / total_races
     ```
   
   - **Recovery Time**: Estimated from win rate
     ```python
     recovery_time = 20 - (win_rate * 10)
     ```
   
   - **Heart Rate**: Estimated from speed
     ```python
     heart_rate = 130 + (speed / speed.max() * 40)
     ```

4. **Categorical Features**
   - **Temperament**: Derived from speed quartiles
     - Categories: ['Calm', 'Energetic', 'Nervous']
     - Used safe quantile cutting to handle duplicates
   
   - **Training Response**: Based on horse_rating quartiles
     - Categories: ['Poor', 'Fair', 'Good', 'Excellent']
     - Handled duplicate values in quantile calculation

## Data Processing Challenges & Solutions

1. **Duplicate Values in Categorical Creation**
   - **Challenge**: `pd.qcut` failing on duplicate values
   - **Solution**: Implemented `safe_qcut` function using rank method
     ```python
     def safe_qcut(series, n, labels):
         try:
             return pd.qcut(series, n, labels=labels)
         except ValueError:
             ranked = series.rank(method='first')
             return pd.qcut(ranked, n, labels=labels)
     ```

2. **Missing Values**
   - Filled using column means for numerical features
   - Maintained data integrity while handling gaps

3. **Unit Conversions**
   - Standardized all measurements to metric system
   - Ensured realistic value ranges for horse characteristics

4. **Feature Scaling**
   - Normalized speed-based features
   - Scaled physical measurements to realistic ranges

## Data Quality Checks

The data processing pipeline includes several quality checks:

1. **Statistical Validation**
   - Verifies reasonable ranges for physical measurements
   - Checks win rate distribution
   - Validates feature correlations

2. **Completeness Checks**
   - Monitors missing value percentages
   - Ensures critical features are populated

3. **Consistency Verification**
   - Cross-references race results
   - Validates temporal sequences 