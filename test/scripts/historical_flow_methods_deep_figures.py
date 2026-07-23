# %% [markdown]
# historical_flow 방법론 심화 문서용 그림 생성기
#
# 목적: (1) 파이프라인을 4개 층위(구간정의/표현/거리/색인)로 분해한 도식,
# (2) 현재 클러스터 가중치가 표준화 부재로 무력화됨(곡선 0.1% vs rsi 지배) 실측,
# (3) 거리 메트릭(유클리드/DTW/상관)이 최근접 이웃을 다르게 뽑음 실증,
# (4) 함수형(FDA) B-스플라인 표현 예시, (5) 유클리드 클러스터 색인이 DTW 이웃을 놓침 실증.
# 축/제목은 CJK 폰트 깨짐 방지로 영어. 실제 마트 데이터 사용.
#
# 실행: PYTHONPATH=. uv run python test/scripts/historical_flow_methods_deep_figures.py

# %%
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from scipy.interpolate import BSpline, make_smoothing_spline

DB = "data/upbit_data.db"
OUT = Path("test/images/historical_flow_methods_deep_20260723")
OUT.mkdir(parents=True, exist_ok=True)
WL = 96
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
rng = np.random.default_rng(0)

FCOLS = [
    "return_total", "realized_vol", "mdd_in_window", "trend_slope_per_hour",
    "value_z_last", "volume_z_last", "rsi_14_last", "roc_16_last",
    "bb_width_20_last", "amihud_illiq_mean",
]


def l1(a, b):
    return abs(float(a) - float(b))


# %%
con = duckdb.connect(DB, read_only=True)
feat = con.execute(
    f"""
    SELECT f.window_id, f.ticker, f.return_path_json, f.factor_vector_json, f.context_vector_json,
           n.archetype_id
    FROM historical_flow_features f
    JOIN (SELECT DISTINCT query_window_id, archetype_id
          FROM historical_flow_neighbors WHERE window_length={WL}) n
      ON n.query_window_id = f.window_id
    WHERE f.window_length={WL} AND f.index_universe='liquid-top'
    LIMIT 4000
    """
).df()
con.close()
paths = np.vstack([json.loads(x) for x in feat["return_path_json"]])
facs = np.vstack([json.loads(x) for x in feat["factor_vector_json"]])
ctxs = np.vstack([json.loads(x) for x in feat["context_vector_json"]])
arch = feat["archetype_id"].to_numpy()
tick = feat["ticker"].to_numpy()
print(f"loaded {len(paths)} windows")

# %% [markdown]
# ## 그림 1 — 파이프라인 4층위 분해와 목적 conflation 지점
# 구간정의(Segmentation) -> 표현(Representation) -> 거리(Distance) -> 색인(Indexing).
# 클러스터는 원래 '구간/기준 정의'로 이해됐지만 코드에선 '색인(속도)'에 들어갔고,
# 거리 층의 metric(유클리드 색인 vs DTW 쿼리)도 어긋난다.

# %%
fig, ax = plt.subplots(figsize=(12, 6.5))
ax.axis("off")
layers = [
    ("1. Segmentation\n(define the 'segment')", "Fixed sliding windows\n(16/48/96/288 bars, stride 8)",
     "Alternatives: change-point (PELT, BOCPD),\nregime-switching (HMM), event-anchored"),
    ("2. Representation\n(encode a segment)", "return path (96-d) + factor stats (10-d)\n+ context (9-d, currently ALL ZERO)",
     "Alternatives: FDA basis/FPCA scores,\nSAX symbols, shapelets, learned embedding"),
    ("3. Distance\n(compare two segments)", "BUILD: Euclidean lockstep\nQUERY: DTW  <-- INCONSISTENT",
     "Alternatives: soft-DTW, correlation/SBD,\ncovariance/spectral, functional L2"),
    ("4. Indexing\n(avoid O(N^2))", "K-means archetypes (Euclidean)\n<-- this is where 'cluster' actually sits",
     "Alternatives: LB_Keogh+UCR pruning,\nk-Shape/DBA, HNSW/IVF ANN"),
]
y0 = 0.9
colors = ["#2e5d8f", "#3b7a57", "#a8541b", "#7a3b8f"]
for i, (title, cur, alt) in enumerate(layers):
    y = y0 - i * 0.22
    ax.add_patch(plt.Rectangle((0.02, y - 0.16), 0.30, 0.19, fc=colors[i], alpha=0.85, ec="k"))
    ax.text(0.17, y - 0.065, title, ha="center", va="center", color="w", fontsize=11, fontweight="bold")
    ax.add_patch(plt.Rectangle((0.35, y - 0.16), 0.30, 0.19, fc="#f0f0f0", ec="k"))
    ax.text(0.50, y - 0.065, "CURRENT:\n" + cur, ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((0.68, y - 0.16), 0.30, 0.19, fc="#fffbe6", ec="k"))
    ax.text(0.83, y - 0.065, alt, ha="center", va="center", fontsize=8.2)
    if i < 3:
        ax.annotate("", xy=(0.17, y - 0.17), xytext=(0.17, y - 0.20 + 0.02),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(0.5, 0.98, "Figure 1. Four layers of 'find similar past segments' — and where the current design conflates them",
        ha="center", fontsize=12, fontweight="bold")
ax.text(0.5, 0.02,
        "Conflation: 'cluster' entered as Layer-1 basis-definition but lives in Layer-4 indexing;  "
        "Layer-3 metric differs between build (Euclidean) and query (DTW).",
        ha="center", fontsize=9, style="italic", color="#a1231b")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(OUT / "fig1_pipeline_layers.png", bbox_inches="tight")
plt.close(fig)
print("saved fig1")

# %% [markdown]
# ## 그림 2 — 클러스터 가중치가 표준화 부재로 무력화 (핵심 싱크 불일치)
# 문서상 shape 0.5 / factor 0.3 / context 0.2 이지만, factor가 표준화 안 된 원시값이라
# rsi_14_last(0~100)가 분산을 독식. 실제 k-means 거리를 좌우하는 블록별 비중을 실측.

# %%
sw, fw, cw = 0.5, 0.3, 0.2
vp = float(np.sum((paths * np.sqrt(sw)).var(axis=0)))
vf = float(np.sum((facs * np.sqrt(fw)).var(axis=0)))
vc = float(np.sum((ctxs * np.sqrt(cw)).var(axis=0)))
tot = vp + vf + vc
fac_var = facs.var(axis=0)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
# 좌: 블록별 실제 기여 vs 의도한 가중치
blocks = ["shape\n(path, 96-d)", "factor\n(stats, 10-d)", "context\n(9-d, all 0)"]
actual = np.array([vp, vf, vc]) / tot * 100
intended = np.array([50, 30, 20])
x = np.arange(3)
axL.bar(x - 0.2, intended, 0.4, label="intended weight (%)", color="#9bbcd6")
axL.bar(x + 0.2, actual, 0.4, label="ACTUAL variance share (%)", color="#c0392b")
for xi, a in zip(x, actual):
    axL.text(xi + 0.2, a + 1.5, f"{a:.1f}%", ha="center", fontsize=9, fontweight="bold")
axL.set_xticks(x); axL.set_xticklabels(blocks)
axL.set_ylabel("Share of k-means distance (%)")
axL.set_title("Intended 0.5/0.3/0.2 vs what actually drives the clustering")
axL.legend()
# 우: factor 블록 내 컬럼 분산
order = np.argsort(fac_var)[::-1]
axR.barh([FCOLS[i] for i in order][::-1], fac_var[order][::-1], color="#a8541b")
axR.set_xscale("log")
axR.set_xlabel("Variance (log scale)")
axR.set_title("Within 'factor': rsi_14_last (0-100 scale) dominates all else")
fig.suptitle("Figure 2. The documented weights are nullified: shape drives ~0.1%, a single RSI value drives clustering", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig2_weights_nullified.png", bbox_inches="tight")
plt.close(fig)
print(f"saved fig2  (shape={actual[0]:.2f}%, factor={actual[1]:.2f}%, context={actual[2]:.2f}%)")

# %% [markdown]
# ## 그림 3 — 거리 메트릭이 '가장 닮은 과거 구간'을 다르게 뽑는다
# 같은 표현(return path)에 대해 Euclidean(lockstep) / DTW / correlation-distance 로
# 최근접 이웃을 각각 뽑아 비교. 메트릭 선택이 결과를 바꾼다.

# %%
def corr_dist(a, B):
    a = a - a.mean()
    Bc = B - B.mean(axis=1, keepdims=True)
    num = Bc @ a
    den = (np.linalg.norm(Bc, axis=1) * np.linalg.norm(a) + 1e-12)
    return 1.0 - num / den


NC = 1500  # 후보 풀
cand = rng.choice(len(paths), NC, replace=False)
cand_paths = paths[cand]
q = int(cand[7])  # 데모용 쿼리 하나
qp = paths[q]

eu = np.linalg.norm(cand_paths - qp, axis=1)
co = corr_dist(qp, cand_paths)
dt = np.array([fastdtw(qp, cp, dist=l1)[0] for cp in cand_paths])
for arr in (eu, co, dt):
    arr[cand == q] = np.inf  # 자기 자신 제외
nn_eu, nn_co, nn_dt = cand[np.argmin(eu)], cand[np.argmin(co)], cand[np.argmin(dt)]

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(qp * 100, color="k", lw=3, label=f"QUERY ({tick[q]})", zorder=5)
ax.plot(paths[nn_eu] * 100, color="#2e5d8f", lw=1.8, label=f"nearest by Euclidean ({tick[nn_eu]})")
ax.plot(paths[nn_co] * 100, color="#3b7a57", lw=1.8, ls="--", label=f"nearest by Correlation ({tick[nn_co]})")
ax.plot(paths[nn_dt] * 100, color="#c0392b", lw=1.8, ls=":", label=f"nearest by DTW ({tick[nn_dt]})")
ax.axhline(0, color="gray", lw=0.7)
ax.set_xlabel("Step within window (0..95)")
ax.set_ylabel("Cumulative log return (%)")
same = len({int(nn_eu), int(nn_co), int(nn_dt)})
ax.set_title(f"Figure 3. Same query, three metrics pick {'DIFFERENT' if same>1 else 'the same'} nearest past window\n"
             "(metric choice, not just data, decides what counts as 'similar')")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig3_metric_disagreement.png", bbox_inches="tight")
plt.close(fig)
print(f"saved fig3  (nn_eu={nn_eu}, nn_co={nn_co}, nn_dt={nn_dt}, distinct={same})")

# %% [markdown]
# ## 그림 4 — 함수형(FDA): 곡선을 매끄러운 함수로 표현
# 노이즈 있는 return path를 평활 스플라인으로 적합해 '함수'로 다룬다. FDA는 이렇게 공통
# 정의역에서 함수/도함수를 비교하거나 소수의 functional PCA 점수로 요약한다.

# %%
qy = paths[q] * 100
t = np.arange(len(qy), dtype=float)
spl = make_smoothing_spline(t, qy, lam=8.0)
smooth = spl(t)
deriv = spl.derivative()(t)  # 속도(순간 기울기) = 국면 정보

fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
a1.plot(t, qy, color="#9aa0a6", lw=1, marker=".", ms=3, label="raw return path (noisy)")
a1.plot(t, smooth, color="#c0392b", lw=2.5, label="smoothing-spline fit f(t)")
a1.axhline(0, color="gray", lw=0.7)
a1.set_ylabel("Cumulative log return (%)")
a1.set_title("Figure 4. Functional view: represent a segment as a smooth function f(t)")
a1.legend(fontsize=9)
a2.plot(t, deriv, color="#2e5d8f", lw=2, label="first derivative f'(t) = local velocity")
a2.axhline(0, color="gray", lw=0.7)
a2.set_xlabel("Step within window (0..95)")
a2.set_ylabel("f'(t)  (%/step)")
a2.set_title("Derivative separates up/down phases; FDA can compare curves via f, f', or FPCA scores")
a2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig4_fda_functional.png", bbox_inches="tight")
plt.close(fig)
print("saved fig4")

# %% [markdown]
# ## 그림 5 — 유클리드 클러스터 색인은 DTW 이웃을 놓친다 (색인/쿼리 메트릭 불일치)
# 다수 쿼리에 대해 진짜 DTW 최근접 이웃을 구하고, 그 이웃이 쿼리와 '같은 k-means archetype'에
# 들어있는 비율을 잰다. 낮으면, 유클리드 클러스터로 후보를 좁히면 DTW 이웃을 놓친다는 뜻.

# %%
NQ = 40
qs = rng.choice(len(paths), NQ, replace=False)
pool = rng.choice(len(paths), NC, replace=False)
pool_paths = paths[pool]
same_cluster_dtwnn = 0
eu_dt_agree = 0
for qi in qs:
    qpp = paths[qi]
    d_dt = np.array([fastdtw(qpp, cp, dist=l1)[0] for cp in pool_paths])
    d_eu = np.linalg.norm(pool_paths - qpp, axis=1)
    mask = pool == qi
    d_dt[mask] = np.inf; d_eu[mask] = np.inf
    nn_dt_i = pool[int(np.argmin(d_dt))]
    nn_eu_i = pool[int(np.argmin(d_eu))]
    if arch[nn_dt_i] == arch[qi]:
        same_cluster_dtwnn += 1
    if nn_dt_i == nn_eu_i:
        eu_dt_agree += 1
frac_missed = 100 * (1 - same_cluster_dtwnn / NQ)
frac_disagree = 100 * (1 - eu_dt_agree / NQ)

fig, ax = plt.subplots(figsize=(8.5, 5.2))
bars = ax.bar(
    ["DTW nearest neighbor\nfalls OUTSIDE query's\nEuclidean k-means cluster",
     "Euclidean-NN and DTW-NN\nare DIFFERENT windows"],
    [frac_missed, frac_disagree],
    color=["#c0392b", "#a8541b"], width=0.6,
)
for b, v in zip(bars, [frac_missed, frac_disagree]):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
ax.set_ylim(0, 105)
ax.set_ylabel("% of queries")
ax.set_title(f"Figure 5. Euclidean k-means index is inconsistent with DTW retrieval\n"
             f"(n={NQ} queries, pool={NC}); high bars = the fast index would MISS true DTW analogs")
fig.tight_layout()
fig.savefig(OUT / "fig5_index_inconsistency.png", bbox_inches="tight")
plt.close(fig)
print(f"saved fig5  (DTW-NN outside cluster={frac_missed:.0f}%, EU/DTW disagree={frac_disagree:.0f}%)")

print("ALL DEEP FIGURES DONE ->", OUT)
