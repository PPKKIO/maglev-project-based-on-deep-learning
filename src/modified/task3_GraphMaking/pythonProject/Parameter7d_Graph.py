import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# ====== 1. 读取你生成好的 Sobol 参数 CSV ======
csv_path = "sobol7d_1024.csv"
df = pd.read_csv(csv_path)

# 明确只取 7 个 Sobol 参数（防止 CSV 里还有别的列）
names = ['k', 'c', 'h0', 'i0', 'kh', 'kv', 'V_kmh']
X = df[names].values   # shape = (1024, 7)

# ====== 2. 参数显示方式与范围 ======
modes = ['log', 'log', 'lin', 'lin', 'lin', 'lin', 'lin']

lo = np.array([3.0e7, 5.0e3, 8e-3, 20.0, 4.0e3, 20.0, 200.0])
hi = np.array([6.0e7, 2.0e4, 12e-3, 30.0, 9.0e3, 80.0, 500.0])

# ====== 3. log 参数处理（k, c 用 log10 展示） ======
Xp = X.copy()
names_plot = names.copy()

for j in range(7):
    if modes[j] == 'log':
        Xp[:, j] = np.log10(X[:, j])
        lo[j] = np.log10(lo[j])
        hi[j] = np.log10(hi[j])
        names_plot[j] = r'$\log_{10}(' + names[j] + r')$'

N = Xp.shape[0]

# ====== 4. 7×7 Sobol Pair Plot ======
fig, axes = plt.subplots(7, 7, figsize=(14, 11))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for r in range(7):
    for c in range(7):
        ax = axes[r, c]

        if r == c:
            ax.hist(Xp[:, c], bins=18, density=True)
            ax.set_xlim(lo[c], hi[c])
            ax.set_yticks([])
        else:
            ax.scatter(
                Xp[:, c], Xp[:, r],
                s=8,
                alpha=0.25
            )
            ax.set_xlim(lo[c], hi[c])
            ax.set_ylim(lo[r], hi[r])
            ax.set_xticks([])
            ax.set_yticks([])

        if r == 6:
            ax.set_xlabel(names_plot[c])
        if c == 0:
            ax.set_ylabel(names_plot[r])

        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(True)

fig.suptitle(
    f"Sobol 7D sampling coverage (N={N})\n"
    f"Parameters loaded from Training_set_MATLAB.csv",
    fontsize=14
)

plt.savefig("sobol_7d_pairplot.png", dpi=300, bbox_inches="tight")
plt.show()
