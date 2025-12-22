import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========= 可改：文件与目标列名 =========
CSV_PATH = "Training_set_MATLAB.csv"
TARGET_COL = "a_z_max (Maximum vertical acceleration)"  # 如果你csv里叫别的，改这里

# ========= 固定划分：717/154/153（按行顺序，不打乱） =========
N_TRAIN, N_VAL = 717, 154  # test 自动是 153

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(42)

class SeqDataset(Dataset):
    def __init__(self, Xseq, y):
        self.X = torch.tensor(Xseq, dtype=torch.float32)
        self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

def to_seq(x2d: np.ndarray) -> np.ndarray:
    # (N, 7) -> (N, 7, 1)
    return x2d.reshape(x2d.shape[0], x2d.shape[1], 1).astype(np.float32)

class LSTMRegressor(nn.Module):
    """
    Sobol 7维静态参数 -> 当作序列长度7 (每步1维) -> 回归输出
    """
    def __init__(self, hidden=96, num_layers=2, bidir=True, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feat_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)        # (B, 7, feat_dim)
        last = out[:, -1, :]         # (B, feat_dim)
        return self.head(last)       # (B, 1)

@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y = np.vstack(ys).ravel()
    p = np.vstack(ps).ravel()
    return y, p

def metrics(y, p):
    rmse = np.sqrt(mean_squared_error(y, p))
    mae  = mean_absolute_error(y, p)
    r2   = r2_score(y, p)
    return rmse, mae, r2

def main():
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=[c for c in df.columns if str(c).lower().startswith("unnamed")], errors="ignore")

    if TARGET_COL not in df.columns:
        cand = [c for c in df.columns if "a_z_max" in str(c)]
        if len(cand) == 1:
            target = cand[0]
        else:
            raise ValueError(f"目标列没找到：{TARGET_COL}；候选：{cand}")
    else:
        target = TARGET_COL

    X_cols = [c for c in df.columns if c != target]
    X = df[X_cols].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)

    assert X.shape[0] == 1024, f"样本数不是1024：{X.shape[0]}"
    print("X shape:", X.shape, "target:", target)

    # 固定切分（不打乱）
    X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
    X_val,   y_val   = X[N_TRAIN:N_TRAIN+N_VAL], y[N_TRAIN:N_TRAIN+N_VAL]
    X_test,  y_test  = X[N_TRAIN+N_VAL:], y[N_TRAIN+N_VAL:]  # <- 确保这行正确

    # 标准化（只用train拟合）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # (N,7)->(N,7,1)
    X_train_seq = to_seq(X_train_s)
    X_val_seq   = to_seq(X_val_s)
    X_test_seq  = to_seq(X_test_s)

    train_loader = DataLoader(SeqDataset(X_train_seq, y_train), batch_size=64, shuffle=True)
    val_loader   = DataLoader(SeqDataset(X_val_seq, y_val), batch_size=256, shuffle=False)
    test_loader  = DataLoader(SeqDataset(X_test_seq, y_test), batch_size=256, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMRegressor(hidden=96, num_layers=2, bidir=True, dropout=0.25).to(device)

    criterion = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    best = float("inf")
    best_state = None
    patience = 50
    wait = 0

    for epoch in range(1, 801):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        vy, vp = predict_all(model, val_loader, device)
        val_rmse, val_mae, val_r2 = metrics(vy, vp)
        sch.step(val_rmse)

        if val_rmse < best - 1e-6:
            best = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 20 == 0:
            lr = opt.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | val RMSE {val_rmse:.6f} MAE {val_mae:.6f} R2 {val_r2:.6f} | lr {lr:.2e}")

        if wait >= patience:
            print("Early stop.")
            break

    model.load_state_dict(best_state)

    # ===== TEST =====
    ty, tp = predict_all(model, test_loader, device)
    test_rmse, test_mae, test_r2 = metrics(ty, tp)
    print("\n==== TEST ====")
    print(f"RMSE={test_rmse:.6f}  MAE={test_mae:.6f}  R2={test_r2:.6f}")
    print("Features:", X_cols)

    # ========= Permutation Importance（测试集置乱重要性 + 权重）=========
    @torch.no_grad()
    def rmse_on_array(Xseq_np, y_np):
        model.eval()
        xb = torch.tensor(Xseq_np, dtype=torch.float32, device=device)
        pred = model(xb).detach().cpu().numpy().ravel()
        return float(np.sqrt(mean_squared_error(y_np, pred)))

    base_rmse = rmse_on_array(X_test_seq, y_test)

    rng = np.random.default_rng(42)
    n_repeats = 30  # 想更稳可改50，越大越慢
    importances = []

    for j, name in enumerate(X_cols):
        deltas = []
        for _ in range(n_repeats):
            Xp = X_test_seq.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, j, 0] = Xp[perm, j, 0]  # 打乱第j个“时间步”(特征)
            rmse_p = rmse_on_array(Xp, y_test)
            deltas.append(rmse_p - base_rmse)
        importances.append(float(np.mean(deltas)))

    imp = np.array(importances, dtype=np.float64)
    imp = np.maximum(imp, 0.0)  # 防止数值抖动出现负值
    weights = imp / (imp.sum() + 1e-12)

    res = pd.DataFrame({
        "feature": X_cols,
        "rmse_increase": imp,
        "weight(sum_to_1)": weights
    }).sort_values("rmse_increase", ascending=False).reset_index(drop=True)

    print("\n==== Permutation Feature Importance (TEST) ====")
    print(f"Baseline TEST RMSE: {base_rmse:.6f}")
    print(res.to_string(index=False))

    # 保存一份，方便你做图/写报告
    res.to_csv("feature_importance_lstm_perm.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: feature_importance_lstm_perm.csv")

    # ========= ⭐ 导出模型 + 预处理器 + 特征顺序 =========
    import joblib, json

    # 1) 保存模型权重（最关键）
    torch.save(best_state, "lstm_state_dict.pt")

    # 2) 保存标准化器（预测时必须用同一个）
    joblib.dump(scaler, "lstm_scaler.pkl")

    # 3) 保存特征顺序（手动输入必须按这个顺序）
    with open("feature_order_lstm.json", "w", encoding="utf-8") as f:
        json.dump(X_cols, f, ensure_ascii=False, indent=2)

    # 4) 保存模型结构超参数（保证预测脚本构建结构一致）
    lstm_meta = {"hidden": 96, "num_layers": 2, "bidir": True, "dropout": 0.25}
    with open("lstm_meta.json", "w", encoding="utf-8") as f:
        json.dump(lstm_meta, f, ensure_ascii=False, indent=2)

    print("\nSaved model artifacts:")
    print("  lstm_state_dict.pt")
    print("  lstm_scaler.pkl")
    print("  feature_order_lstm.json")
    print("  lstm_meta.json")


if __name__ == "__main__":
    main()


