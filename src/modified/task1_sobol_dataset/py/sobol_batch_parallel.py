import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import math
import multiprocessing as mp
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ===== 参数范围（Stage 1 核心7维） =====
RANGES: Dict[str, Tuple[str, float, float]] = {
    "k":    ("log", 3.0e7, 6.0e7),
    "c":    ("log", 5.0e3, 2.0e4),
    "h0":   ("lin", 8e-3, 12e-3),
    "i0":   ("lin", 20.0, 30.0),
    "kh":   ("lin", 4.0e3, 9.0e3),
    "kv":   ("lin", 20.0, 80.0),
    "V_kmh":("lin", 200.0, 500.0),
}

# 固定参数（调试可改为小网格: s=2,e=10,num_dt=20）
FIXED = {
    "mc": 39000, "Jc": 2e6, "l": 1.548, "mm": 1000,
    "L": 25, "mg": 6000, "EI": 8.58e10, "kb": 3.2e10, "kc": 1e7,
    "s": 8, "e": 50, "num_dt": 100
}

# ===== Sobol 采样 =====
def map_col(u, mode, lo, hi):
    return lo + (hi - lo) * u if mode == "lin" else lo * (hi / lo) ** u

def sobol_df(N: int, ranges=RANGES) -> pd.DataFrame:
    """固定顺序的 Sobol 样本，并附带 sid（0..N-1）"""
    keys = list(ranges.keys())
    d = len(keys)
    m = int(math.log2(N))
    if 2 ** m != N:
        N = 2 ** m
        print(f"[info] 将 N 调整为 2^m = {N}")
    sampler = qmc.Sobol(d=d, scramble=True)
    U = sampler.random_base2(m)  # 行顺序固定，可用于断点续跑
    data = {k: map_col(U[:, j], *ranges[k]) for j, k in enumerate(keys)}
    df = pd.DataFrame(data)
    df.insert(0, "sid", np.arange(len(df), dtype=int))
    return df

def diag_guard_ok(p: Dict[str, float], mc=39000, mm=1000) -> bool:
    g = 9.81
    F0 = (mc + 8 * mm) * g / 8.0
    K0 = F0 / ((p["i0"] / p["h0"]) ** 2)
    Ci = 2 * K0 * p["i0"] / (p["h0"] ** 2)
    Ch = 2 * K0 * (p["i0"] ** 2) / (p["h0"] ** 3)
    return (p["k"] + Ci * p["kh"] - Ch) > 0.0

# ===== 进程内 Runtime（只初始化一次）=====
_APP = None
def _worker_init():
    global _APP
    import maglev_sim
    _APP = maglev_sim.initialize()
def _worker_fini():
    global _APP
    if _APP is not None:
        try: _APP.terminate()
        except: pass
        _APP = None

def _run_one(payload: Dict[str, float]) -> Dict:
    global _APP
    sid = int(payload.pop("sid"))
    base = {"sid": sid, **payload}  # 所有输入参数都带回
    try:
        res = _APP.solve_model4_fem_pkg(json.dumps({**FIXED, **payload}))
        ok = bool(res["ok"])
        out = {"ok": ok, "err_code": int(res["err_code"])}
        if ok:
            feats = res["features"]
            out.update({
                "a_car_peak": float(feats["a_car_peak"]),
                "a_car_rms":  float(feats["a_car_rms"]),
                "label_unstable": int(feats["label_unstable"]),
                "message": ""   # 成功也给空字符串，保证列存在
            })
        else:
            out.update({
                "a_car_peak": np.nan,
                "a_car_rms":  np.nan,
                "label_unstable": np.nan,
                "message": str(res.get("message",""))
            })
        return {**base, **out}
    except Exception as e:
        return {**base,
                "ok": False, "err_code": 999,
                "a_car_peak": np.nan, "a_car_rms": np.nan, "label_unstable": np.nan,
                "message": f"{e}"}


def main(N=128, outfile="../sobol_dataset.csv", procs: Optional[int]=None, flush_every: int=10, chunk_size: int=1):
    # 1) 生成样本 + 断点续跑（读取已完成 sid）
    df = sobol_df(N)
    mask = df.apply(lambda r: diag_guard_ok(r.drop(labels=["sid"]).to_dict()), axis=1)
    df = df[mask].reset_index(drop=True)

    done_sids = set()
    if os.path.exists(outfile):
        try:
            existed = pd.read_csv(outfile, usecols=["sid"])
            done_sids = set(int(x) for x in existed["sid"].dropna().astype(int).tolist())
            print(f"[resume] 已存在 {len(done_sids)} 条，按 sid 跳过这些样本")
        except Exception as e:
            print(f"[resume] 读取现有文件失败，将重新创建：{e}")
            os.remove(outfile)

    todo = df[~df["sid"].isin(done_sids)].reset_index(drop=True)
    total = len(df); remain = len(todo)
    print(f"[info] 有效样本 {total} 条；待计算 {remain} 条")

    if remain == 0:
        print(f"[done] 无需计算，{outfile} 已包含全部 {total} 条。")
        return

    # 2) 并行配置
    if procs is None:
        half = max(1, mp.cpu_count() // 2)
        procs = min(6, half)  # 保守默认，避免内存爆
    print(f"[info] 启动 {procs} 个进程并行（flush_every={flush_every}）")

    # 3) 创建/确认 CSV 头
    cols = ["sid"] + [k for k in RANGES.keys()] + ["ok","err_code","a_car_peak","a_car_rms","label_unstable","message"]
    if not os.path.exists(outfile):
        pd.DataFrame(columns=cols).to_csv(outfile, index=False, encoding="utf-8-sig")

    written = 0
    with mp.Pool(processes=procs, initializer=_worker_init) as pool:
        buf: List[Dict] = []
        for i, res in enumerate(pool.imap_unordered(_run_one, todo.to_dict(orient="records"), chunksize=chunk_size), 1):
            buf.append(res)
            if len(buf) >= flush_every:
                pd.DataFrame(buf)[cols].to_csv(outfile, mode="a", header=False, index=False, encoding="utf-8-sig")
                written += len(buf)
                buf.clear()
                if i % (flush_every*5) == 0:
                    print(f"[progress] 已追加 {written}/{remain} 条")

        if buf:
            chunk = pd.DataFrame(buf).reindex(columns=cols)
            chunk.to_csv(outfile, mode="a", header=False, index=False, encoding="utf-8-sig")
            written += len(buf)

    _worker_fini()
    print(f"[done] 本次新增 {written} 条；文件：{outfile}")
    print(f"[hint] 下次续跑会自动跳过 sid 已完成的样本")

if __name__ == "__main__":
    # 例：N=512 正式采样；保守并发（自动=CPU/2 上限6）。需要更快可 procs=6。
    # 调试可：FIXED.update({"s":2,"e":10,"num_dt":20})
    main(N=512, outfile="../sobol_dataset.csv", procs=None, flush_every=10)
