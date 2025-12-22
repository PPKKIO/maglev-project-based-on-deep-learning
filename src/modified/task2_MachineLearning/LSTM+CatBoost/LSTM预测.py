import json
import numpy as np
import joblib
import torch
import torch.nn as nn

def to_seq(x2d: np.ndarray) -> np.ndarray:
    return x2d.reshape(x2d.shape[0], x2d.shape[1], 1).astype(np.float32)

class LSTMRegressor(nn.Module):
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
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)

def main():
    # 1) 载入特征顺序 + meta + scaler + 权重
    with open("feature_order_lstm.json", "r", encoding="utf-8") as f:
        X_cols = json.load(f)

    with open("lstm_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    scaler = joblib.load("lstm_scaler.pkl")
    state = torch.load("lstm_state_dict.pt", map_location="cpu")

    # 2) 构建模型并加载参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMRegressor(**meta).to(device)
    model.load_state_dict(state)
    model.eval()

    # 3) 按顺序输入参数
    print("请按顺序输入 7 个参数：")
    for i, name in enumerate(X_cols, 1):
        print(f"{i}. {name}")

    vals = []
    for name in X_cols:
        vals.append(float(input(f"{name} = ")))

    x = np.array(vals, dtype=np.float32).reshape(1, -1)

    # 4) 标准化 + 变形为 (1, 7, 1)
    x_s = scaler.transform(x)
    x_seq = to_seq(x_s)

    # 5) 预测
    with torch.no_grad():
        xb = torch.tensor(x_seq, dtype=torch.float32, device=device)
        y_pred = model(xb).cpu().numpy().ravel()[0]

    print("\n预测最大垂向加速度 =", float(y_pred))

if __name__ == "__main__":
    main()
