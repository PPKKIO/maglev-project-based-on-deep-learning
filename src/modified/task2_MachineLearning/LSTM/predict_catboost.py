import json
import numpy as np
import joblib
from catboost import CatBoostRegressor

# 载入
with open("feature_order.json", "r", encoding="utf-8") as f:
    X_cols = json.load(f)

scaler = joblib.load("catboost_scaler.pkl")

model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

print("按顺序输入 7 个参数：")
print(X_cols)

vals = []
for name in X_cols:
    vals.append(float(input(f"{name} = ")))

x = np.array(vals, dtype=np.float64).reshape(1, -1)
x_s = scaler.transform(x)

y_pred = model.predict(x_s)[0]
print("\n预测最大垂向加速度 =", y_pred)
