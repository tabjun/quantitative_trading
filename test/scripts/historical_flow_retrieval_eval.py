# %% [markdown]
# historical_flow 검색 품질 평가 (목적 정합 평가지표)
#
# 핵심 질문: "과거 유사국면(이웃)의 *이후* 전개가 현재 국면의 *이후* 전개를 예측하는가?"
# 연구 목적(방향=무신호, 크기/변동성=유신호, MDD=생존제약)에 맞춰 세 축을 평가하고,
# **랜덤 이웃 baseline**과 대조해 '유사도 검색이 실제로 정보를 더하는지'를 본다.
#
# 평가 지표:
#  - 크기(변동성): analog 평균 |forward_return_24h| 로 쿼리 실제 |forward_return_24h| 순위예측(Spearman IC)
#  - MDD(생존): analog 평균 future_mdd_3d 로 쿼리 future_mdd_3d 순위예측(Spearman IC)
#  - 방향(sanity): analog 평균 수익률 부호로 쿼리 부호 적중률(≈50% 기대, EDA와 정합)
#  각 축을 실제 이웃 vs 랜덤 이웃으로 비교.
#
# 실행: PYTHONPATH=. uv run python test/scripts/historical_flow_retrieval_eval.py

# %%
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DB = "data/upbit_data.db"
OUT_IMG = Path("test/images/historical_flow_eval_20260723")
OUT_IMG.mkdir(parents=True, exist_ok=True)
TOP_K = 10
rng = np.random.default_rng(0)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def evaluate(window_length: int) -> dict:
    con = duckdb.connect(DB, read_only=True)
    nb = con.execute(
        f"""
        SELECT query_window_id, neighbor_window_id, rank
        FROM historical_flow_neighbors
        WHERE window_length = {window_length} AND rank <= {TOP_K}
        """
    ).df()
    ev = con.execute(
        f"""
        SELECT window_id, forward_return_24h, future_mdd_3d
        FROM historical_flow_event_stats
        WHERE window_length = {window_length}
        """
    ).df()
    con.close()

    ev = ev.dropna(subset=["forward_return_24h", "future_mdd_3d"])
    ret = dict(zip(ev["window_id"], ev["forward_return_24h"]))
    mdd = dict(zip(ev["window_id"], ev["future_mdd_3d"]))

    nb = nb[nb["neighbor_window_id"].isin(ret) & nb["query_window_id"].isin(ret)]
    nb["nb_ret"] = nb["neighbor_window_id"].map(ret)
    nb["nb_mdd"] = nb["neighbor_window_id"].map(mdd)

    grp = nb.groupby("query_window_id")
    agg = grp.agg(
        pred_mag=("nb_ret", lambda s: np.mean(np.abs(s))),
        pred_dir=("nb_ret", lambda s: np.sign(np.mean(s))),
        pred_mdd=("nb_mdd", "mean"),
        n_neigh=("nb_ret", "size"),
    ).reset_index()
    agg = agg[agg["n_neigh"] >= 3]
    agg["act_ret"] = agg["query_window_id"].map(ret)
    agg["act_mag"] = agg["act_ret"].abs()
    agg["act_mdd"] = agg["query_window_id"].map(mdd)
    agg = agg.dropna()

    # 랜덤 이웃 baseline: 이웃 결과를 쿼리에 무작위로 재배정
    all_nb_ret = nb["nb_ret"].to_numpy()
    all_nb_mdd = nb["nb_mdd"].to_numpy()
    n_q = len(agg)
    rand_mag = np.array([np.mean(np.abs(rng.choice(all_nb_ret, TOP_K))) for _ in range(n_q)])
    rand_mdd = np.array([np.mean(rng.choice(all_nb_mdd, TOP_K)) for _ in range(n_q)])

    def ic(a, b):
        r, _ = spearmanr(a, b)
        return float(r)

    mag_ic = ic(agg["pred_mag"], agg["act_mag"])
    mag_ic_rand = ic(rand_mag, agg["act_mag"].to_numpy())
    mdd_ic = ic(agg["pred_mdd"], agg["act_mdd"])
    mdd_ic_rand = ic(rand_mdd, agg["act_mdd"].to_numpy())
    dir_hit = float(np.mean(np.sign(agg["act_ret"]) == agg["pred_dir"]))

    return {
        "window_length": window_length,
        "n_queries": n_q,
        "mag_ic": mag_ic,
        "mag_ic_rand": mag_ic_rand,
        "mdd_ic": mdd_ic,
        "mdd_ic_rand": mdd_ic_rand,
        "dir_hit": dir_hit,
        "agg": agg,
    }


# %%
results = {wl: evaluate(wl) for wl in (16, 48, 96, 288)}
summary = pd.DataFrame(
    [{k: v for k, v in r.items() if k != "agg"} for r in results.values()]
)
print("=== 검색 품질 요약 (실제 이웃 vs 랜덤 이웃) ===")
print(summary.to_string(index=False))

# %% [markdown]
# ## 그림 1 — 크기(변동성) 예측력: analog 평균 |24h 수익률| vs 실제 |24h 수익률|
# %%
R96 = results[96]
agg = R96["agg"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
a1.scatter(agg["pred_mag"] * 100, agg["act_mag"] * 100, s=6, alpha=0.25, color="#2e5d8f")
a1.set_xlabel("Analog-implied magnitude: mean |24h return| of k analogs (%)")
a1.set_ylabel("Actual |24h return| after query (%)")
a1.set_title(f"Magnitude predictability (wl=96)\nSpearman IC = {R96['mag_ic']:.3f} (random analogs: {R96['mag_ic_rand']:.3f})")
lim = np.nanpercentile(agg["act_mag"] * 100, 99)
a1.set_xlim(0, lim); a1.set_ylim(0, lim)

labels = ["magnitude\n(|24h ret|)", "MDD\n(3d)"]
real_ics = [R96["mag_ic"], R96["mdd_ic"]]
rand_ics = [R96["mag_ic_rand"], R96["mdd_ic_rand"]]
x = np.arange(2)
a2.bar(x - 0.2, real_ics, 0.4, label="real analogs", color="#2e7d32")
a2.bar(x + 0.2, rand_ics, 0.4, label="random analogs", color="#b0b0b0")
for xi, v in zip(x - 0.2, real_ics):
    a2.text(xi, v + 0.005 * np.sign(v or 1), f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
a2.axhline(0, color="k", lw=0.8)
a2.set_xticks(x); a2.set_xticklabels(labels)
a2.set_ylabel("Spearman rank IC (analog forecast vs actual)")
a2.set_title(f"Retrieval adds signal iff real >> random\n(direction hit-rate = {R96['dir_hit']*100:.1f}%, ~50% expected)")
a2.legend()
fig.suptitle("Figure. historical_flow retrieval quality — purpose-aligned (magnitude/MDD predictable, direction not)", y=1.02)
fig.tight_layout()
fig.savefig(OUT_IMG / "eval_predictability.png", bbox_inches="tight")
plt.close(fig)
print("saved eval_predictability.png")

# csv 저장
summary.to_csv(OUT_IMG / "eval_summary.csv", index=False)
print("saved eval_summary.csv")
print("DONE")
