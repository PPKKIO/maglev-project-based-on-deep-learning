import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ========= 可改 =========
CSV_PATH = "Training_set_MATLAB.csv"
TARGET_COL = "a_z_max (Maximum vertical acceleration)"

# ========= 固定划分 =========
N_TRAIN, N_VAL = 717, 154   # test = 153

# ========= 1) 读数据 =========
df = pd.read_csv(CSV_PATH)
df = df.drop(columns=[c for c in df.columns if str(c).lower().startswith("unnamed")],
             errors="ignore")

if TARGET_COL not in df.columns:
    cand = [c for c in df.columns if "a_z_max" in str(c)]
    if len(cand) == 1:
        target = cand[0]
    else:
        raise ValueError(f"目标列没找到：{TARGET_COL}；候选：{cand}")
else:
    target = TARGET_COL

X_cols = [c for c in df.columns if c != target]

X = df[X_cols].to_numpy(dtype=np.float64)
y = df[target].to_numpy(dtype=np.float64)

assert X.shape[0] == 1024, f"样本数不是1024：{X.shape[0]}"
print("X shape:", X.shape, "target:", target)

# ========= 2) 固定切分（不打乱） =========
X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
X_val,   y_val   = X[N_TRAIN:N_TRAIN+N_VAL], y[N_TRAIN:N_TRAIN+N_VAL]
X_test,  y_test  = X[N_TRAIN+N_VAL:], y[N_TRAIN+N_VAL:]

# ========= 3) （可选）标准化 =========
# CatBoost 对是否标准化不敏感，但你前面模型都做了，这里保持一致
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ========= 4) CatBoost 模型 =========
model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    early_stopping_rounds=100,
    verbose=200
)

# ========= 5) 训练 =========
model.fit(
    X_train_s, y_train,
    eval_set=(X_val_s, y_val),
    use_best_model=True
)

# ========= 6) 测试集评估 =========
y_pred = model.predict(X_test_s)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n==== TEST ====")
print(f"RMSE={rmse:.6f}  MAE={mae:.6f}  R2={r2:.6f}")

# ========= 7) 特征重要性（CatBoost 内置，基于损失变化） =========
importances = model.get_feature_importance(type="PredictionValuesChange")

# 归一化为权重（和为1）
importances = np.maximum(importances, 0.0)
weights = importances / (importances.sum() + 1e-12)

res = pd.DataFrame({
    "feature": X_cols,
    "importance": importances,
    "weight(sum_to_1)": weights
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\n==== Feature Importance (CatBoost) ====")
print(res.to_string(index=False))

# ========= 8) 保存为 CSV =========
res.to_csv("feature_importance_catboost.csv",
           index=False,
           encoding="utf-8-sig")

print("\nSaved: feature_importance_catboost.csv")

import joblib, json

# 保存模型（CatBoost原生格式）
model.save_model("catboost_model.cbm")

# 保存scaler（你训练时用的StandardScaler）
joblib.dump(scaler, "catboost_scaler.pkl")

# 保存特征顺序（非常关键：手动输入必须按这个顺序）
with open("feature_order.json", "w", encoding="utf-8") as f:
    json.dump(X_cols, f, ensure_ascii=False, indent=2)

print("Saved: catboost_model.cbm, catboost_scaler.pkl, feature_order.json")
