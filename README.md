[README.txt](https://github.com/user-attachments/files/28218967/README.txt)
README

Spatio-Temporal Supply and Demand Optimization for Ride-Hailing Services

Project Overview

This project proposes a spatio-temporal machine learning framework for
improving supply-demand balance in ride-hailing systems. The framework
combines demand prediction and optimization techniques using NYC Yellow
Taxi trip data.

Objectives

-   Predict short-term ride-hailing demand
-   Compare multiple machine learning models
-   Analyze feature importance
-   Reduce supply-demand imbalance through optimization
-   Evaluate optimization using shortage reduction and relocation cost

Dataset

NYC TLC Yellow Taxi Trip Data (2025)

Required columns: - tpep_pickup_datetime - PULocationID - DOLocationID

Machine Learning Models

-   Random Forest
-   Decision Tree
-   Linear Regression
-   KNN
-   XGBoost

Evaluation Metrics: - MAE - RMSE

Optimization

Supply-demand gap:

Gap = Predicted Demand − Estimated Supply

Optimization method: Integer Programming vehicle rebalancing

Additional operational metric: Relocation Cost

Relocation Cost Formula:

Relocation Cost = ΣΣ xij × cij

where: - xij = relocated vehicles - cij = movement cost between zones

Results

Example results:

-   Best model: XGBoost
-   RMSE: 36.19
-   Shortage reduction: 6.86%
-   Vehicles relocated: 1410
-   Relocation cost: 65.76

Required Libraries

pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install xgboost
pip install pulp

Run Project

python capstone.py

Future Work

-   Include weather and traffic information
-   Explore GRU/LSTM models
-   Improve optimization strategy
-   Develop a real-time deployment framework

Author: Zeyi Wang
