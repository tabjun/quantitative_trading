# %% [markdown]
# historical_flow 마트 방법론 설명용 그림 생성기
#
# 목적: "과거 유사 국면 검색"을 (1) 기존 전수비교 방식과 (2) 클러스터 기반 방식으로
# 어떻게 수행하는지, 그리고 DTW(시계열 정렬) vs 유클리드(격자 대응)의 차이를
# 실제 마트 데이터로 시각화한다. 축/제목은 CJK 폰트 깨짐 방지를 위해 영어로 쓴다.
#
# 실행: PYTHONPATH=. uv run python test/scripts/historical_flow_method_figures.py

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

DB_PATH = "data/upbit_data.db"
OUT_DIR = Path("test/images/historical_flow_method_20260723")
OUT_DIR.mkdir(parents=True, exist_ok=True)
WL = 96  # 설명용 창 길이 (24시간 = 96 x 15분)

plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def load_paths(limit: int = 4000) -> tuple[np.ndarray, list[str], dict[str, int]]:
    con = duckdb.connect(DB_PATH, read_only=True)
    feats = con.execute(
        f"""
        SELECT window_id, return_path_json
        FROM historical_flow_features
        WHERE window_length = {WL} AND index_universe = 'liquid-top'
        ORDER BY window_id
        LIMIT {limit}
        """
    ).df()
    arch = con.execute(
        f"""
        SELECT DISTINCT query_window_id AS window_id, archetype_id
        FROM historical_flow_neighbors
        WHERE window_length = {WL}
        """
    ).df()
    con.close()
    arch_map = dict(zip(arch["window_id"], arch["archetype_id"]))
    paths = np.vstack([json.loads(p) for p in feats["return_path_json"]])
    ids = list(feats["window_id"])
    return paths, ids, arch_map


paths, ids, arch_map = load_paths()
print(f"loaded {len(paths)} windows of length {paths.shape[1]}")

# %% [markdown]
# ## 그림 1 — window 하나가 무엇인가: 연속 곡선(누적 로그수익률 경로)
# 당신 표현대로 "2차원 평면 위 함수 형태의 연속 그래프"가 맞다. 각 window는
# 길이 96(24시간)짜리 누적 로그수익률 곡선이다.

# %%
fig, ax = plt.subplots(figsize=(9, 5))
for i in range(6):
    ax.plot(paths[i] * 100, lw=1.6, alpha=0.85, label=ids[i].split("|")[0])
ax.axhline(0, color="k", lw=0.8, alpha=0.5)
ax.set_xlabel("Step within window (15-min bars, 0..95 = 24h)")
ax.set_ylabel("Cumulative log return (%)")
ax.set_title("Figure 1. One 'window' = a continuous return-path curve (length 96)")
ax.legend(fontsize=9, ncol=3)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_window_as_curve.png")
plt.close(fig)
print("saved fig1")

# %% [markdown]
# ## 그림 2 — DTW(시계열 정렬) vs 유클리드(격자 대응)
# 같은 모양이지만 시간축으로 밀린/늘어난 두 곡선을 비교. 유클리드는 i번째-i번째를
# 그대로 잇는 '격자 대응'이라 시간이 밀리면 거리가 부풀려진다. DTW는 최적 정렬 경로로
# 모양이 비슷하면 시간이 밀려도 낮은 거리를 준다.

# %%
# 시간 왜곡 예시를 위해: 한 실제 곡선을 골라 '같은 모양을 시간축으로 늘린' 변형을 만든다.
# np.interp로 매끄러운 비선형 시간 스트레치 -> 모양은 동일, 시간만 왜곡.
base = paths[0]
L = len(base)
orig_t = np.linspace(0.0, 1.0, L)
stretch_t = orig_t**1.35  # 비선형 시간 왜곡(앞부분은 느리게, 뒷부분은 빠르게)
warped = np.interp(orig_t, stretch_t, base)

# 공정 비교: 둘 다 '점별 절대차의 합'(L1) 기준으로 통일한다.
# lockstep(대각 경로)은 DTW가 고를 수 있는 여러 경로 중 하나이므로, DTW <= lockstep 이 보장된다.
eu_dist = float(np.sum(np.abs(base - warped)))
dtw_dist, dtw_path = fastdtw(base, warped, dist=lambda a, b: abs(float(a) - float(b)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
off = 0.02  # 시각화용 수직 오프셋
for ax, title, pairing in [
    (axes[0], f"Lockstep (i-to-i, L1)  dist={eu_dist:.3f}", [(i, i) for i in range(L)]),
    (axes[1], f"DTW (warped alignment, L1)  dist={dtw_dist:.3f}", dtw_path),
]:
    ax.plot(base, color="#1f4e79", lw=2, label="window A")
    ax.plot(warped + off, color="#c0392b", lw=2, label="window B (shifted/stretched)")
    for a, b in pairing[:: max(1, len(pairing) // 40)]:
        ax.plot([a, b], [base[a], warped[b] + off], color="gray", lw=0.6, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Step within window")
    ax.legend(fontsize=9)
axes[0].set_ylabel("Cumulative log return")
fig.suptitle("Figure 2. Same shape, warped in time: lockstep inflates distance, DTW does not (same L1 basis)", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_dtw_vs_euclidean.png", bbox_inches="tight")
plt.close(fig)
print(f"saved fig2  (euclidean={eu_dist:.3f}, dtw={dtw_dist:.3f})")

# %% [markdown]
# ## 그림 3 — 클러스터(국면 원형)가 실제로 무엇을 묶는가
# K-means가 만든 archetype 중 4개를 골라, 그 안에 속한 window들의 곡선을 겹쳐 그린다.
# 같은 클러스터의 곡선들이 '모양 가족'(상승형/하락형/V자/급등 등)으로 묶이는지 눈으로 확인.

# %%
labels = np.array([arch_map.get(wid, -1) for wid in ids])
valid = labels >= 0
# 표본이 충분한 상위 4개 클러스터 선택
uniq, counts = np.unique(labels[valid], return_counts=True)
top4 = uniq[np.argsort(counts)[::-1][:4]]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, cl in zip(axes.ravel(), top4):
    members = paths[labels == cl]
    sample = members[: min(60, len(members))]
    for m in sample:
        ax.plot(m * 100, color="#5b9bd5", lw=0.6, alpha=0.25)
    ax.plot(members.mean(axis=0) * 100, color="#c0392b", lw=2.5, label="cluster mean")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_title(f"Archetype #{cl}  (n={len(members)})")
    ax.legend(fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("Step within window")
for ax in axes[:, 0]:
    ax.set_ylabel("Cumulative log return (%)")
fig.suptitle("Figure 3. What each K-means archetype groups: families of return-path shapes", y=1.0)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_archetypes.png", bbox_inches="tight")
plt.close(fig)
print(f"saved fig3  (clusters shown: {list(top4)})")

# %% [markdown]
# ## 그림 4 — 계산 방식 차이: 전수비교 O(N^2) vs 클러스터 내부 비교
# 왼쪽: 모든 쌍을 서로 비교(브루트포스). 오른쪽: 먼저 K개 원형으로 묶고, 같은 원형 안에서만
# 비교. 화살표(비교) 개수가 확 줄어드는 것이 속도 이득의 핵심.

# %%
rng_pts = 12
np.random.seed(0)  # 개념 도식용 좌표 (실데이터 아님, 배치 시각화 목적)
xy = np.random.rand(rng_pts, 2)
groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
colors = ["#1f4e79", "#c0392b", "#2e7d32"]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
# 왼쪽: 전수비교
ax = axes[0]
for i in range(rng_pts):
    for j in range(i + 1, rng_pts):
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color="gray", lw=0.4, alpha=0.4)
ax.scatter(xy[:, 0], xy[:, 1], c="#1f4e79", s=90, zorder=3)
n_pairs_brute = rng_pts * (rng_pts - 1) // 2
ax.set_title(f"Before: compare ALL pairs (brute force)\n{n_pairs_brute} comparisons for {rng_pts} points  ->  O(N^2)")
ax.set_xticks([]); ax.set_yticks([])
# 오른쪽: 클러스터 내부만
ax = axes[1]
n_pairs_cluster = 0
for g in range(3):
    idx = np.where(groups == g)[0]
    for a_i in range(len(idx)):
        for b_i in range(a_i + 1, len(idx)):
            i, j = idx[a_i], idx[b_i]
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=colors[g], lw=0.7, alpha=0.6)
            n_pairs_cluster += 1
    # 클러스터 경계 표시
    cx, cy = xy[idx, 0].mean(), xy[idx, 1].mean()
    ax.scatter([cx], [cy], marker="*", s=400, c=colors[g], edgecolor="k", zorder=4)
for g in range(3):
    idx = groups == g
    ax.scatter(xy[idx, 0], xy[idx, 1], c=colors[g], s=90, zorder=3)
ax.set_title(f"After: cluster first (star=archetype), compare only WITHIN cluster\n{n_pairs_cluster} comparisons  ->  ~O(N^2 / K)")
ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Figure 4. Why it got faster: brute-force all-pairs vs cluster-scoped comparison", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_compute_schematic.png", bbox_inches="tight")
plt.close(fig)
print(f"saved fig4  (brute={n_pairs_brute}, cluster={n_pairs_cluster})")

# %% [markdown]
# ## 그림 5 — 클러스터가 '실제로' 무엇을 기준으로 갈라지는가
# 그림 3에서 곡선 모양은 뚜렷이 안 갈렸다. 80개 archetype을 (변동성, 총수익률) 평면에 찍으면,
# 클러스터가 '모양'이 아니라 변동성×수익률 국면(regime)을 타일처럼 덮는다는 게 드러난다.
# 즉 현재 K-means는 시계열 모양 정렬(DTW)이 아니라 요약통계 국면 분리에 가깝다.

# %%
con2 = duckdb.connect(DB_PATH, read_only=True)
cstats = con2.execute(
    f"""
    SELECT n.archetype_id, COUNT(*) AS n,
           AVG(f.realized_vol) AS vol, AVG(f.return_total) AS ret
    FROM (SELECT DISTINCT query_window_id, archetype_id
          FROM historical_flow_neighbors WHERE window_length={WL}) n
    JOIN historical_flow_features f
      ON f.window_id = n.query_window_id AND f.window_length={WL}
    GROUP BY n.archetype_id
    """
).df()
con2.close()

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(
    cstats["vol"] * 100,
    cstats["ret"] * 100,
    s=np.sqrt(cstats["n"]) * 8,
    c=cstats["ret"] * 100,
    cmap="RdYlGn",
    edgecolor="k",
    linewidth=0.5,
    alpha=0.85,
)
ax.axhline(0, color="k", lw=0.8, alpha=0.5)
ax.set_xlabel("Cluster mean realized volatility (%)")
ax.set_ylabel("Cluster mean total return (%)")
ax.set_title(
    "Figure 5. Each dot = one archetype (size ~ #windows).\n"
    "Clusters tile the volatility x return REGIME plane, not crisp visual shapes."
)
fig.colorbar(sc, ax=ax, label="mean total return (%)")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig5_cluster_regime.png", bbox_inches="tight")
plt.close(fig)
print("saved fig5")

print("ALL FIGURES DONE ->", OUT_DIR)
