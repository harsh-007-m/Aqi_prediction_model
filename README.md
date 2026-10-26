# AQI Prediction Model 
Predicting the AQI of Indian cities using linear regression

## Overview
This project process real-world environmental monitoring data to forecast air quality. It handles data cleaning, feature engineering, one-hot encoding and trains a Linear Regression Model. The model is used to predict AQI across 10 major cities using environmental pollutant levels and temporal data.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, Joblib
* **Model:** Linear Regression

## Dataset
The Dataset was taken from Kaggle and was initially named "india_city_aqi_2015_2023". It contained **32870** records from Jan 2015 to Dec 2023. It contains attributes such as:
* City
* Date
* pm25
* pm10
* no2
* so2
* co
* o3
* aqi
* aqi_category

## Data Preprocessing and Feature Extraction
- **Datetime Conversion:** Converted the `Date` object column into a proper Pandas datetime format.
- **Temporal Extraction:** Extracted new features including `Year`, `Month`, `Day`, and `Day_of_the_Week`.
- **Data Audit:** Inspected for missing and duplicate values and verified feature ranges; the dataset was thoroughly pre-cleaned, requiring minimal raw cleanup.
- **Categorical Encoding:** One-hot encoded the `city` column using `pd.get_dummies()` (with `drop_first=True`) to prevent multicollinearity and handle regional baseline shifts.

## Insights and Findings  
**Unusually High Accuracy:** The model achieved an exceptional $R^2$ score of **0.9999** during evaluation. 
- **The Underlying Formula:** By inspecting the model's feature coefficients, we discovered exact weights (e.g., ~0.5 for PM2.5, ~0.3 for PM10, ~0.1 for O3 and NO2). This revealed that the dataset's AQI values were synthetically derived from a direct, clean linear combination of the pollutant features rather than random natural variance, allowing the Linear Regression model to successfully reverse-engineer the governing math formula.
- 
## Results
* **Accuracy:** Achieved an $R^2$ score of **0.9999** and a minimal Mean Squared Error (MSE) of **0.08**.
* **Feature Importance Analysis:** Uncovered that particulate matter (`PM2.5` and `PM10`) serves as the primary driver behind AQI calculations.
* **Geospatial Handling:** Successfully encoded 10 unique cities using Pandas one-hot encoding (`drop_first=True`) to account for regional baseline pollution differences.

## Repository Structure
* `aqi_prediction_notebook.ipynb`: Complete code covering EDA, preprocessing, and training.
* `aqi_forecast_model.pkl`: Serialized production-ready model saved via Joblib.
