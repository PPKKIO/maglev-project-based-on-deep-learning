import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.gridspec import GridSpec

# ====== 读取数据 ======
csv_path = "sobol7d_1024.csv"
df = pd.read_csv(csv_path)

cols  = ['k','c','h0','i0','kh','kv','V_kmh']
modes = ['log','log','lin','lin','lin','lin','lin']

lo = np.array([3.0e7, 5.0e3, 8e-3, 20.0, 4.0e3, 20.0, 200.0])
hi = np.array([6.0e7, 2.0e4, 12e-3, 30.0, 9.0e3, 80.0, 500.0])

X = df[cols].to_numpy()
N = X.shape[0]

# ====== log 参数处理 ======
Xp = X.copy()
cols_plot = cols.copy()
lo_plot = lo.copy()
hi_plot = hi.copy()

for j in range(7):
    if modes[j] == 'log':
        Xp[:, j] = np.log10(X[:, j])
        lo_plot[j] = np.log10(lo[j])
        hi_plot[j] = np.log10(hi[j])
        cols_plot[j] = f'log10({cols[j]})'

# ====== 布局：4 行 × 2 列（右列最后一行留空）======
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(
    nrows=4,
    ncols=2,
    figure=fig,
    width_ratios=[1, 1],
    hspace=0.35,   # 行间距（全局）
    wspace=0.25    # 列间距
)

rng = np.random.default_rng(42)

# ====== 左列：4 个 ======
for i in range(4):
    ax = fig.add_subplot(gs[i, 0])
    y = rng.uniform(-0.5, 0.5, size=N)
    ax.scatter(Xp[:, i], y, s=10, alpha=0.25)
    ax.set_xlim(lo_plot[i], hi_plot[i])
    ax.set_yticks([])
    ax.set_title(f"{cols_plot[i]}  (N={N})", fontsize=10)
    # ax.text(0.01, 0.90,
    #         f"range: [{lo[i]:.3g}, {hi[i]:.3g}]",
    #         transform=ax.transAxes, fontsize=8, va='top')

# ====== 右列：3 个（占前 3 行，第 4 行自然空出）======
for j in range(3):
    idx = j + 4
    ax = fig.add_subplot(gs[j, 1])
    y = rng.uniform(-0.5, 0.5, size=N)
    ax.scatter(Xp[:, idx], y, s=10, alpha=0.25)
    ax.set_xlim(lo_plot[idx], hi_plot[idx])
    ax.set_yticks([])
    ax.set_title(f"{cols_plot[idx]}  (N={N})", fontsize=10)
    # ax.text(0.01, 0.90,
    #         f"range: [{lo[idx]:.3g}, {hi[idx]:.3g}]",
    #         transform=ax.transAxes, fontsize=8, va='top')

fig.suptitle(
    "7D Sobol parameter strip plots\n(y-jitter is visualization only)",
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("sobol_7d_stripplots_2col.png", dpi=300, bbox_inches="tight")
plt.show()
