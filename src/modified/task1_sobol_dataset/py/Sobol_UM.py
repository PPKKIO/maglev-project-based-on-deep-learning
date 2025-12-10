import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol
from typing import Dict, Tuple

# 7 维 Sobol 参数范围
RANGES_UM: Dict[str, Tuple[str, float, float]] = {
    "kz_1":       ("log", 0.6 * 1.18e7,  2.0 * 1.18e7),   # [7.08e6, 2.36e7]
    "cz_1":       ("log", 0.5 * 5.0e4,   2.0 * 5.0e4),    # [2.5e4, 1.0e5]
    "kz_2":       ("log", 0.5 * 1.5e5,   2.0 * 1.5e5),    # [7.5e4, 3.0e5]
    "cz_2":       ("log", 0.5 * 1.0e3,   2.0 * 1.0e3),    # [5.0e2, 2.0e3]
    "m_frame":    ("lin", 900.0,         1500.0),
    "m_carboday": ("lin", 2.0e4,         5.0e4),
    "v0":         ("lin", 200.0,         600.0),          # km/h
}

N_SAMPLES = 512   # 2^m，自己改

def gen_sobol_samples(ranges: Dict[str, Tuple[str, float, float]], n_samples: int) -> pd.DataFrame:
    names = list(ranges.keys())
    d = len(names)

    m = int(np.ceil(np.log2(n_samples)))
    n = 2 ** m

    sampler = Sobol(d=d, scramble=True)
    u = sampler.random_base2(m=m)[:n_samples]  # [0,1) 样本

    X = np.zeros_like(u)
    for j, name in enumerate(names):
        mode, lo, hi = ranges[name]
        if mode == "lin":
            X[:, j] = lo + (hi - lo) * u[:, j]
        elif mode == "log":
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            X[:, j] = 10 ** (log_lo + (log_hi - log_lo) * u[:, j])
        else:
            raise ValueError(f"unknown mode {mode}")

    return pd.DataFrame(X, columns=names)

if __name__ == "__main__":
    df = gen_sobol_samples(RANGES_UM, N_SAMPLES)
    df.to_csv("sobol_inputs_um_7d.csv", index=False)
    print(df.head())
