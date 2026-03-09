# Ride-Hailing Demand Prediction and Rebalancing

## Project Overview
This project predicts taxi demand using the NYC Yellow Taxi dataset and analyzes supply-demand imbalance.

The system aggregates taxi trip data into hourly demand by taxi zone and trains a Random Forest model to predict next-hour demand.

## Dataset
Dataset: NYC TLC Yellow Taxi Trip Records

File used:
yellow_tripdata_2025-10.csv

Source:
NYC Taxi and Limousine Commission

## Implementation Steps
1. Load taxi trip dataset
2. Construct hourly demand dataset
3. Feature engineering
4. Train Random Forest model
5. Evaluate model using MAE and RMSE
6. Analyze supply-demand imbalance
7. Simulate vehicle rebalancing

## Results
Key outputs include:

- Hourly demand dataset
- Model performance (MAE and RMSE)
- Imbalance distribution
- Rebalancing simulation

Screenshots are provided in the screenshots folder.

## How to Run
1. Open the notebook `capstone_implementation.ipynb`
2. Upload dataset file `yellow_tripdata_2025-10.csv`
3. Run all cells sequentially.

## Libraries
- pandas
- numpy
- scikit-learn