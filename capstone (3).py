#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# =========================
# 1. Load ALL 2025 monthly datasets
# =========================
data_path = os.path.expanduser("~/Documents")

file_list = [
    f"yellow_tripdata_2025-{str(i).zfill(2)}.csv"
    for i in range(1, 13)
]

# 只读取后续分析需要的列，减少内存占用
use_columns = ["tpep_pickup_datetime", "PULocationID", "DOLocationID"]

df_list = []

for file in file_list:
    file_path = os.path.join(data_path, file)

    if os.path.exists(file_path):
        print(f"Loading {file}...")
        temp_df = pd.read_csv(file_path, usecols=use_columns)
        df_list.append(temp_df)
    else:
        print(f"Warning: {file} not found, skipped.")

if len(df_list) == 0:
    raise FileNotFoundError("No monthly CSV files were found in the specified path.")

# 合并全年数据
df = pd.concat(df_list, ignore_index=True)

print("\nAll available 2025 data loaded successfully!")


# In[2]:


# =========================
# 2. Basic preprocessing
# =========================
print("\nConverting pickup datetime...")
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")

df = df.dropna(subset=["tpep_pickup_datetime", "PULocationID", "DOLocationID"])
df["PULocationID"] = df["PULocationID"].astype(int)
df["DOLocationID"] = df["DOLocationID"].astype(int)

print("Rows after cleaning:", len(df))


# In[3]:


# =========================
# 3. Build hourly demand dataset
# =========================
print("\nBuilding hourly demand dataset...")

df["date"] = df["tpep_pickup_datetime"].dt.date
df["hour"] = df["tpep_pickup_datetime"].dt.hour
df["pickup_time"] = df["tpep_pickup_datetime"].dt.floor("H")

# 每个区域每小时的 demand = pickup 次数
hourly_demand = (
    df.groupby(["pickup_time", "PULocationID"])
    .size()
    .reset_index(name="demand")
)

print("Hourly demand dataset created!")
print(hourly_demand.head())
print("Total hourly records:", len(hourly_demand))


# In[4]:


# =========================
# 4. Feature engineering
# =========================
print("\nCreating features...")

hourly_demand["hour"] = hourly_demand["pickup_time"].dt.hour
hourly_demand["day_of_week"] = hourly_demand["pickup_time"].dt.dayofweek
hourly_demand["month"] = hourly_demand["pickup_time"].dt.month
hourly_demand["day"] = hourly_demand["pickup_time"].dt.day
hourly_demand["is_weekend"] = hourly_demand["day_of_week"].isin([5, 6]).astype(int)

# 按区域排序，做滞后特征
hourly_demand = hourly_demand.sort_values(["PULocationID", "pickup_time"])

# lag features
hourly_demand["lag_1"] = hourly_demand.groupby("PULocationID")["demand"].shift(1)
hourly_demand["lag_2"] = hourly_demand.groupby("PULocationID")["demand"].shift(2)
hourly_demand["lag_3"] = hourly_demand.groupby("PULocationID")["demand"].shift(3)

# rolling mean features
hourly_demand["rolling_mean_3"] = (
    hourly_demand.groupby("PULocationID")["demand"]
    .shift(1)
    .rolling(window=3)
    .mean()
)

hourly_demand["rolling_mean_6"] = (
    hourly_demand.groupby("PULocationID")["demand"]
    .shift(1)
    .rolling(window=6)
    .mean()
)

# target: 预测下一小时 demand
hourly_demand["target"] = hourly_demand.groupby("PULocationID")["demand"].shift(-1)

# 删除因为 shift / rolling 产生的空值
hourly_demand = hourly_demand.dropna().reset_index(drop=True)

print("Feature engineering completed!")
print(hourly_demand.head())
print("Rows after feature engineering:", len(hourly_demand))


# In[5]:


# =========================
# 5. Save engineered dataset
# =========================
output_engineered = os.path.join(data_path, "hourly_demand_dataset_2025.csv")
hourly_demand.to_csv(output_engineered, index=False)
print(f"\nEngineered dataset saved to: {output_engineered}")


# In[6]:


# =========================
# 6. Prepare X and y
# =========================
feature_columns = [
    "PULocationID",
    "hour",
    "day_of_week",
    "month",
    "day",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_mean_6"
]

X = hourly_demand[feature_columns]
y = hourly_demand["target"]

# 单独保存时间列，供后面画图使用
time_series = hourly_demand["pickup_time"]


# In[7]:


hourly_demand.groupby("hour")["demand"].mean().plot(figsize=(8,5))
plt.xlabel("Hour of Day")
plt.ylabel("Average Demand")
plt.title("Average Hourly Demand by Hour")
plt.grid(True)
plt.show()


# In[8]:


# =========================
# 7. Train-test split
# =========================
# 注意：时间序列最好不要 random shuffle，这里按时间顺序切分更合理
split_index = int(len(hourly_demand) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 对应测试集的时间
time_test = time_series.iloc[split_index:].reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# In[9]:


# =========================
# 8. Standardize data for KNN
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 9. Define evaluation function
# =========================
def evaluate_model(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }


# In[10]:


# =========================
# 10. Train 5 models
# =========================
results = []

# -------- 1. Random Forest --------
print("\nTraining Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results.append(evaluate_model("Random Forest", y_test, rf_pred))

# -------- 2. Decision Tree --------
print("Training Decision Tree...")
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
results.append(evaluate_model("Decision Tree", y_test, dt_pred))

# -------- 3. XGBoost --------
print("Training XGBoost...")
xgb = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
results.append(evaluate_model("XGBoost", y_test, xgb_pred))

# -------- 4. Linear Regression --------
print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
results.append(evaluate_model("Linear Regression", y_test, lr_pred))

# -------- 5. KNN --------
print("Training KNN...")
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
results.append(evaluate_model("KNN", y_test, knn_pred))


# In[11]:


# =========================
# Actual vs Predicted Demand Plot (Time-based)
# =========================
print("\nPlotting Actual vs Predicted Demand (XGBoost)...")

n = 100   # 你想显示前多少个点

# 取前 n 个时间点和对应结果
plot_time = time_test.iloc[:n]
plot_actual = y_test.iloc[:n]
plot_pred = xgb_pred[:n]

plt.figure(figsize=(14, 6))

plt.plot(plot_time, plot_actual, label="Actual Demand")
plt.plot(plot_time, plot_pred, label="Predicted Demand")

plt.xlabel("Time", fontsize=18)
plt.ylabel("Demand", fontsize=18)
plt.title("Actual vs Predicted Demand (XGBoost)", fontsize=20, fontweight="bold")
plt.legend(fontsize=16)

plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True)
plt.tight_layout()

model_result_path = os.path.join(data_path, "actual_vs_predicted_xgb_time.png")
plt.savefig(model_result_path, dpi=300)
plt.show()

print(f"Actual vs Predicted plot saved to: {model_result_path}")


# In[12]:


# =========================
# City-level Actual vs Predicted Demand Plot
# =========================
print("\nPlotting city-level actual vs predicted demand...")

# 测试集结果表
test_plot_df = hourly_demand.iloc[split_index:].copy().reset_index(drop=True)
test_plot_df["Actual"] = y_test.values
test_plot_df["Predicted"] = xgb_pred

# 按 pickup_time 聚合
city_plot_df = test_plot_df.groupby("pickup_time")[["Actual", "Predicted"]].sum().reset_index()

# 只取前 n 个小时，避免太密
n = 168   # 一周 = 168小时
plot_df = city_plot_df.iloc[:n].copy()

plt.figure(figsize=(14, 6))

plt.plot(plot_df["pickup_time"], plot_df["Actual"], label="Actual Demand", linewidth=2)
plt.plot(plot_df["pickup_time"], plot_df["Predicted"], label="Predicted Demand", linewidth=2)

plt.xlabel("Pickup Time", fontsize=18)
plt.ylabel("Total Demand", fontsize=18)
plt.title("Actual vs Predicted City-level Demand in the first week (XGBoost)", fontsize=20, fontweight="bold")
plt.legend(fontsize=14)
plt.grid(True)

plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()

plot_path = os.path.join(data_path, "city_level_actual_vs_predicted_xgb.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"Plot saved to: {plot_path}")


# In[13]:


# =========================
# 11. Results table
# =========================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="RMSE").reset_index(drop=True)

print("\nModel Comparison Results:")
print(results_df)

output_results = os.path.join(data_path, "model_comparison_results_2025.csv")
results_df.to_csv(output_results, index=False)
print(f"\nModel comparison results saved to: {output_results}")


# In[14]:


# =========================
# 12. Plot model comparison (MAE & RMSE only)
# =========================
plt.figure(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.35

plt.bar(x - width/2, results_df["MAE"], width, label="MAE")
plt.bar(x + width/2, results_df["RMSE"], width, label="RMSE")

plt.xticks(x, results_df["Model"], rotation=20, fontsize=18)
plt.yticks(fontsize=13)

plt.ylabel("Prediction Error (MAE / RMSE)", fontsize=18)

plt.title("Comparison of MAE and RMSE Across Models (NYC Taxi Demand, 2025)", 
          fontsize=15, fontweight='bold')

plt.legend(prop={'size': 12})

plt.title("Comparison of MAE and RMSE Across Models", 
          fontsize=20, fontweight='bold')

plt.tight_layout()

plot_path = os.path.join(data_path, "model_comparison_2025.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"Model comparison plot saved to: {plot_path}")


# In[15]:


# =========================
# 14. Plot Random Forest feature importance
# =========================
feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nRandom Forest Feature Importance:")
print(feature_importance)

output_importance = os.path.join(data_path, "rf_feature_importance_2025.csv")
feature_importance.to_csv(output_importance, index=False)
print(f"Feature importance saved to: {output_importance}")

plt.figure(figsize=(10, 6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance (NYC 2025)")
plt.gca().invert_yaxis()
plt.tight_layout()

importance_plot_path = os.path.join(data_path, "rf_feature_importance_2025.png")
plt.savefig(importance_plot_path, dpi=300)
plt.show()

print(f"Feature importance plot saved to: {importance_plot_path}")


# In[16]:


# =========================
# 15. Save actual vs predicted for best model (RF)
# =========================
prediction_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted_XGB": xgb_pred
})
prediction_output = os.path.join(data_path, "rf_predictions_2025.csv")
prediction_df.to_csv(prediction_output, index=False)

print(f"\nRF predictions saved to: {prediction_output}")
print("\nDone! All steps completed successfully.")


# In[17]:


# =========================
# 16. Ablation Study (based on XGBoost)
# =========================

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# 统一评估函数
def evaluate_ablation(variant_name, modify_content, X_train_var, X_test_var, y_train_var, y_test_var):
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_train_var, y_train_var)
    y_pred_var = model.predict(X_test_var)

    mae = mean_absolute_error(y_test_var, y_pred_var)
    mse = mean_squared_error(y_test_var, y_pred_var)
    rmse = np.sqrt(mse)

    return {
        "Model variant": variant_name,
        "modify content": modify_content,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }

# 目标变量
y_ablation = hourly_demand["target"]

# 你的完整特征
full_features = [
    "PULocationID",
    "hour",
    "day_of_week",
    "month",
    "day",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_mean_6"
]

# 各种 variant
variant_dict = {
    "Completed Model": {
        "modify content": "None",
        "features": full_features
    },
    "Variant 1": {
        "modify content": "Remove the lag feature",
        "features": [f for f in full_features if f not in ["lag_1", "lag_2", "lag_3"]]
    },
    "Variant 2": {
        "modify content": "Remove rolling mean",
        "features": [f for f in full_features if f not in ["rolling_mean_3", "rolling_mean_6"]]
    },
    "Variant 3": {
        "modify content": "Remove the time feature",
        "features": [f for f in full_features if f not in ["hour", "day_of_week", "month", "day", "is_weekend"]]
    },
    "Variant 4": {
        "modify content": "Simplified model (only hour)",
        "features": ["hour"]
    }
}

# 跑 ablation
ablation_results = []

for variant_name, config in variant_dict.items():
    feature_list = config["features"]

    X_ablation = hourly_demand[feature_list]

    # 保持和你主实验一致：按时间顺序切分
    X_train_var = X_ablation.iloc[:split_index]
    X_test_var = X_ablation.iloc[split_index:]
    y_train_var = y_ablation.iloc[:split_index]
    y_test_var = y_ablation.iloc[split_index:]

    result = evaluate_ablation(
        variant_name,
        config["modify content"],
        X_train_var,
        X_test_var,
        y_train_var,
        y_test_var
    )
    ablation_results.append(result)

# 转表
ablation_df = pd.DataFrame(ablation_results)

print("\nAblation Study Results:")
print(ablation_df)

# 保存结果
ablation_output = os.path.join(data_path, "ablation_results_2025.csv")
ablation_df.to_csv(ablation_output, index=False)
print(f"\nAblation results saved to: {ablation_output}")

# =========================
# 17. Plot Ablation Study RMSE
# =========================
plt.figure(figsize=(10, 6))
plt.bar(ablation_df["Model variant"], ablation_df["RMSE"])
plt.ylabel("RMSE")
plt.title("Ablation Study on NYC 2025 Taxi Demand Prediction")
plt.xticks(rotation=20)
plt.tight_layout()

ablation_plot_path = os.path.join(data_path, "ablation_rmse_2025.png")
plt.savefig(ablation_plot_path, dpi=300)
plt.show()

print(f"Ablation plot saved to: {ablation_plot_path}")


# In[18]:


# =========================
# 16. Imports
# =========================
import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# =========================
# 17. Build Supply-Demand State
# =========================
test_state_df = hourly_demand.iloc[split_index:].copy().reset_index(drop=True)

test_state_df["predicted_demand"] = xgb_pred
test_state_df["estimated_supply"] = test_state_df["lag_1"]

# gap
test_state_df["gap"] = test_state_df["predicted_demand"] - test_state_df["estimated_supply"]

# =========================
# 18. Adaptive threshold (STD)
# =========================
std_gap = test_state_df["gap"].std()
gap_threshold = 0.5*std_gap

print(f"STD-based threshold: {gap_threshold:.4f}")

test_state_df["shortage"] = np.where(
    test_state_df["gap"] > gap_threshold,
    np.ceil(test_state_df["gap"]).astype(int),
    0
)

test_state_df["surplus"] = np.where(
    test_state_df["gap"] < -gap_threshold,
    np.ceil(-test_state_df["gap"]).astype(int),
    0
)


# =========================
# 19. Build REAL Transition Probability T_ij
# =========================
print("\nBuilding transition probability T_ij...")

df["hour"] = df["tpep_pickup_datetime"].dt.hour

od_counts = (
    df.groupby(["hour", "PULocationID", "DOLocationID"])
    .size()
    .reset_index(name="trip_count")
)

origin_total = (
    od_counts.groupby(["hour", "PULocationID"])["trip_count"]
    .sum()
    .reset_index(name="total")
)

od_counts = od_counts.merge(origin_total, on=["hour", "PULocationID"])

od_counts["prob"] = od_counts["trip_count"] / od_counts["total"]

transition_dict = {
    (int(r.hour), int(r.PULocationID), int(r.DOLocationID)): r.prob
    for _, r in od_counts.iterrows()
}

def get_T(hour, i, j):
    return transition_dict.get((int(hour), int(i), int(j)), 0)


# =========================
# 20. Relocation Cost (proxy)
# =========================
MAX_ZONE = 262

def cost(i, j):
    return abs(int(i) - int(j)) / MAX_ZONE


# =========================
# 21. Integer Programming per time slot
# =========================
def optimize_slot(slot_df, alpha=1, beta=0.3, gamma=0.5):

    hour = int(slot_df["hour"].iloc[0])

    shortage_df = slot_df[slot_df["shortage"] > 0]
    surplus_df = slot_df[slot_df["surplus"] > 0]

    if len(shortage_df) == 0 or len(surplus_df) == 0:
        return None, None

    shortage = dict(zip(shortage_df["PULocationID"], shortage_df["shortage"]))
    surplus = dict(zip(surplus_df["PULocationID"], surplus_df["surplus"]))

    prob = pulp.LpProblem("Rebalancing", pulp.LpMaximize)

    x = {}
    for i in surplus:
        for j in shortage:
            x[(i,j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat="Integer")

    # objective
    prob += pulp.lpSum(
        x[(i,j)] * (
            alpha * shortage[j]
            - beta * cost(i,j)
            + gamma * get_T(hour, i, j)
        )
        for i in surplus
        for j in shortage
    )

    # constraints
    for i in surplus:
        prob += pulp.lpSum(x[(i,j)] for j in shortage) <= surplus[i]

    for j in shortage:
        prob += pulp.lpSum(x[(i,j)] for i in surplus) <= shortage[j]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    rows = []
    total_move = 0

    for (i,j), var in x.items():
        v = int(var.value()) if var.value() else 0
        if v > 0:
            total_move += v
            rows.append({
                "from": i,
                "to": j,
                "vehicles": v,
                "T_ij": get_T(hour, i, j),
                "cost": cost(i,j)
            })

    return pd.DataFrame(rows), total_move


# =========================
# 22. Run optimization
# =========================
print("\nRunning optimization...")

slots = sorted(test_state_df["pickup_time"].unique())[:168]

plans = []
summary = []

for t in slots:
    slot_df = test_state_df[test_state_df["pickup_time"] == t]

    plan, moved = optimize_slot(slot_df)

    if plan is not None:
        plan["time"] = t
        plans.append(plan)

    summary.append({
        "time": t,
        "before_shortage": slot_df["shortage"].sum(),
        "before_surplus": slot_df["surplus"].sum(),
        "vehicles_moved": moved if moved else 0
    })

plan_df = pd.concat(plans) if plans else pd.DataFrame()
summary_df = pd.DataFrame(summary)


# =========================
# 23. Evaluation
# =========================
before = summary_df["before_shortage"].sum()
moved = summary_df["vehicles_moved"].sum()
after = before - moved

reduction = moved / before * 100 if before > 0 else 0

# Relocation cost evaluation
if not plan_df.empty:
    plan_df["relocation_cost"] = plan_df["vehicles"] * plan_df["cost"]
    total_relocation_cost = plan_df["relocation_cost"].sum()
    average_relocation_cost = total_relocation_cost / moved if moved > 0 else 0
else:
    total_relocation_cost = 0
    average_relocation_cost = 0

print("\nOptimization Result:")
print(f"Before shortage: {before}")
print(f"After shortage: {after}")
print(f"Reduction: {reduction:.2f}%")
print(f"Vehicles moved: {moved}")

print("\nRelocation Cost Evaluation:")
print(f"Total relocation cost: {total_relocation_cost:.4f}")
print(f"Average relocation cost per vehicle: {average_relocation_cost:.4f}")


# =========================
# 24. Plot
# =========================
plt.figure(figsize=(6,4))

plt.bar(["Without Optimization", "With Optimization"], [before, after])

# 标题：加粗 + 放大
plt.title("Shortage Reduction", fontsize=20, fontweight='bold')

# 坐标轴标题：放大
plt.xlabel("Scenario", fontsize=18)
plt.ylabel("Total Shortage", fontsize=18)

# x轴刻度：放大
plt.xticks(fontsize=18)

# y轴刻度：放大
plt.yticks(fontsize=18)

plt.tight_layout()
plt.show()


# In[19]:


plt.figure(figsize=(8,5))  # 原来是(6,4)，加宽！

plt.bar(["Without Optimization", "With Optimization"], [before, after])

# 标题：加粗 + 放大
plt.title("Shortage Reduction", fontsize=20, fontweight='bold')

# 坐标轴标题：放大
plt.xlabel("Scenario", fontsize=18)
plt.ylabel("Total Shortage", fontsize=18)

# x轴刻度：放大
plt.xticks(fontsize=18)

# y轴刻度：放大
plt.yticks(fontsize=18)

plt.tight_layout()
plt.show()


# In[20]:


# =========================
# Build OD transition probability
# =========================

df["hour"] = df["tpep_pickup_datetime"].dt.hour

od_hour_counts = (
    df.groupby(["hour", "PULocationID", "DOLocationID"])
    .size()
    .reset_index(name="trip_count")
)

origin_hour_total = (
    od_hour_counts.groupby(["hour", "PULocationID"])["trip_count"]
    .sum()
    .reset_index(name="origin_total")
)

od_hour_counts = od_hour_counts.merge(
    origin_hour_total,
    on=["hour", "PULocationID"]
)

od_hour_counts["prob"] = (
    od_hour_counts["trip_count"] / od_hour_counts["origin_total"]
)


# In[21]:


# =========================
# Model Result Plot: Actual vs Predicted
# =========================
plt.figure(figsize=(10, 5))

n = 265

plt.plot(y_test.values[:n], label="Actual Demand")
plt.plot(xgb_pred[:n], label="Predicted Demand")

plt.xlabel("Sample Index",fontsize=18)
plt.ylabel("Demand", fontsize=18)

plt.title("Actual vs Predicted Demand (XGBoost)", fontweight='bold',fontsize=20)

plt.legend(fontsize=18)

plt.grid(True)
plt.tight_layout()

model_result_path = os.path.join(data_path, "actual_vs_predicted_xgb.png")
plt.savefig(model_result_path, dpi=300)
plt.show()


# In[22]:


# =========================
# Optimization Comparison Plot with Reduction Annotation
# =========================
plt.figure(figsize=(6, 5))

values = [before, after]
labels = ["Before", "After"]

bars = plt.bar(labels, values)

plt.ylabel("Total Shortage")
plt.title("Optimization Effect on Supply-Demand Shortage")

# 标注数值
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom"
    )

# 标注 reduction
plt.text(
    0.5,
    max(values) * 0.9,
    f"Reduction = {reduction:.2f}%",
    ha="center",
    fontsize=12
)

plt.tight_layout()

optimization_annotated_path = os.path.join(data_path, "optimization_effect_annotated.png")
plt.savefig(optimization_annotated_path, dpi=300)
plt.show()

print(f"Annotated optimization plot saved to: {optimization_annotated_path}")


# In[26]:


# =========================
# Relocation Cost Evaluation Plot
# =========================

plt.figure(figsize=(8,5))

bars = plt.bar(
    ["Relocation Cost"],
    [total_relocation_cost]
)

plt.ylabel("Cost", fontsize=16)

plt.title(
    "Total Relocation Cost After Optimization",
    fontsize=18,
    fontweight='bold'
)

# 在柱子上显示具体数值
for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=12
    )

plt.tight_layout()

relocation_cost_plot_path = os.path.join(
    data_path,
    "relocation_cost_evaluation.png"
)

plt.savefig(relocation_cost_plot_path, dpi=300)
plt.show()

print(f"Relocation cost evaluation plot saved to: {relocation_cost_plot_path}")


# In[ ]:




