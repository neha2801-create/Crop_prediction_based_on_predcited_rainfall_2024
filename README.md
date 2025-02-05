# Crop_prediction_based_on_predicted_rainfall_2024
This project involved prediction of rainfall level for 2024 year (location used: India) using model1 further this predicted rainfall levels for 2024 is fed into model 2 as test data for getting predicted crop levels for 2024.


### Model Architecture
1. **Model 1 (Random Forest Regressor)**
   - Input: Historical rainfall data (2018-2023)
   - Output: Predicted rainfall levels for 2024
   - Features: Grid Search optimization for hyperparameters

2. **Model 2 (Random Forest Classifier)** 
   - Input: Predicted 2024 rainfall data from Model 1
   - Output: Seasonal crop predictions for 2024
   - Features: Season-based segmentation

## Data Sources
- Rainfall Data: Visual Crossing Weather Data Services
- Crop Data: Kaggle dataset (augmented with climate and fertilizer data)

## Key Features
- Historical rainfall pattern analysis (2018-2023)
- Seasonal segmentation for Indian agricultural contexts
- Hyperparameter tuning using Grid Search
- Integration of real-time weather data

## Model Performance
- Model 1 evaluated using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R-squared (R2) score
