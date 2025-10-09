import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import qmc
import maglev_sim


# ==== 1) 参数范围（Stage 1：7维核心） ====
#   mode: "lin" 或 "log"
RANGES: Dict[str, Tuple[str, float, float]] = {
    "k":    ("log", 3.0e7, 6.0e7),
    "c":    ("log", 5.0e3, 2.0e4),
    "h0":   ("lin", 8e-3,  12e-3),
    "i0":   ("lin", 20.0,  30.0),
    "kh":   ("lin", 4.0e3, 9.0e3),
    "kv":   ("lin", 20.0,  80.0),
    "V_kmh":("lin", 200.0, 500.0),
}

# 固定参数（不进 Sobol），可按需改
FIXED = {
    "mc": 39000, "Jc": 2e6, "l": 1.548, "mm": 1000,
    "L": 25, "mg": 6000, "EI": 8.58e10, "kb": 3.2e10, "kc": 1e7,
    "s": 8, "e": 50, "num_dt": 100
}


# ==== 2) 工具函数 ====
def map_col(u: np.ndarray, mode: str, lo: float, hi: float) -> np.ndarray:
    """把 [0,1] 映射到物理范围；log 用几何插值"""
    if mode == "lin":
        return lo + (hi - lo) * u
    # mode == "log"
    return lo * (hi / lo) ** u


def sobol_samples(N: int, ranges=RANGES) -> pd.DataFrame:
    """生成 N 个 Sobol 样本（N 取 2^m 效果最佳）"""
    keys = list(ranges.keys())
    d = len(keys)
    # 如果 N 不是 2^m，就取最近的 2^m
    m = int(math.log2(N)) if abs(N - 2 ** round(math.log2(N))) < 1e-9 else int(math.log2(2 ** round(math.log2(N))))
    N_eff = 2 ** m
    if N_eff != N:
        print(f"[info] Sobol 要求 2^m，已将 N={N} 调整为 {N_eff}")
    sampler = qmc.Sobol(d=d, scramble=True)
    U = sampler.random_base2(m)
    cols = {}
    for j, k in enumerate(keys):
        mode, lo, hi = ranges[k]
        cols[k] = map_col(U[:, j], mode, lo, hi)
    return pd.DataFrame(cols)


def diag_guard_ok(p: Dict[str, float], mc=39000, mm=1000) -> bool:
    """
    硬约束：k + Ci*kh - Ch > 0
    其中：
      F0=(mc+8*mm)g/8
      K0=F0/(i0/h0)^2
      Ci=2*K0*i0/h0^2
      Ch=2*K0*i0^2/h0^3
    """
    g = 9.81
    F0 = (mc + 8 * mm) * g / 8.0
    K0 = F0 / ((p["i0"] / p["h0"]) ** 2)
    Ci = 2 * K0 * p["i0"] / (p["h0"] ** 2)
    Ch = 2 * K0 * (p["i0"] ** 2) / (p["h0"] ** 3)
    return (p["k"] + Ci * p["kh"] - Ch) > 0.0


def run_one(app, params: Dict[str, float]) -> Dict:
    """调用已编译的 MATLAB 包；返回统一字典"""
    # 组装输入（合并固定项；MATLAB 端支持 V_kmh，会自动转 V）
    payload = {**FIXED, **params}
    try:
        res = app.solve_model4_fem_pkg(json.dumps(payload))
        ok = bool(res["ok"])
        out = {
            "ok": ok,
            "err_code": int(res["err_code"]),
        }
        if ok:
            feats = res["features"]
            out.update({
                "a_car_peak": float(feats["a_car_peak"]),
                "a_car_rms":  float(feats["a_car_rms"]),
                "label_unstable": int(feats["label_unstable"]),
            })
        else:
            out["message"] = str(res.get("message", ""))
        return out
    except Exception as e:
        return {"ok": False, "err_code": 999, "message": f"py-exception: {e}"}


# ==== 3) 主流程 ====
def main(N=128, outfile_csv="../sobol_dataset.csv"):
    # 1) 生成 Sobol 样本
    df = sobol_samples(N)

    # 2) 过滤不满足硬约束的样本（必要时补点）
    mask = df.apply(lambda r: diag_guard_ok(r.to_dict()), axis=1)
    df_ok = df[mask].reset_index(drop=True)
    dropped = len(df) - len(df_ok)
    if dropped:
        print(f"[warn] 因硬约束丢弃 {dropped} 个样本（k + Ci*kh - Ch <= 0）")

    # 3) 初始化 MATLAB Runtime（只初始化一次，循环复用）
    app = maglev_sim.initialize()

    # 4) 批量调用
    rows: List[Dict] = []
    for i, rec in df_ok.iterrows():
        params = rec.to_dict()
        r = run_one(app, params)
        r.update(params)  # 把输入参数也带回
        rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"[progress] {i+1}/{len(df_ok)}")

    app.terminate()

    # 5) 保存结果
    out = pd.DataFrame(rows)
    out.to_csv(outfile_csv, index=False, encoding="utf-8-sig")
    print(f"[done] 保存到 {outfile_csv} ，总计 {len(out)} 行；成功 {out['ok'].sum()} 条，失败 {(~out['ok']).sum()} 条")


if __name__ == "__main__":
    # 改 N 就行；建议先 64/128 试跑，通了再 512/1024
    main(N=128, outfile_csv="../sobol_dataset_128.csv")
