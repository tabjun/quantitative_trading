# %% [markdown]
# # 27e번: 변동성 국면별 결과의 입력 통제 점검(시드 5개, 재학습 없음)
#
# 27d는 요동 국면(Q5)의 1시간·4시간·12시간 최선이 순환 신경망(LSTM·GRU)이고, 순환 신경망 묶음이 병렬 딥러닝
# 묶음보다 요동 국면에서 유의하게 낫다고 보고했다. 그런데 27d의 GRU·LSTM은 15분봉 96개를 읽고, 병렬 딥러닝은
# H분 블록 로그 RV 이력을 읽는다. 27c는 이 입력 차이를 전체 기간에서만 통제했고 국면별로는 보지 않았다.
# 이 회차는 27c의 블록 입력 GRU·LSTM(병렬 딥러닝과 같은 입력·표본, 시드 0~4) 예측을 27d와 같은 국면으로 나눠
# 두 가지를 확인한다.
#
# 1. **묶음 비교가 입력을 맞춰도 남는가**: 블록 입력 순환망 − 병렬 딥러닝, 블록 입력 순환망 − 병렬 특징 기반,
#    15분봉 구성 − 블록 구성 순환망(입력·표본·배치가 함께 다른 두 구성의 차이)을 국면 × 예측 구간 30칸에서 검정한다.
# 2. **"요동 국면 최선 = 순환 신경망"이 입력을 맞춰도 남는가**: 블록 입력 GRU·LSTM을 후보에 더해 국면 최선 검정을
#    다시 하고, 순환 신경망 네 모델(15분봉·블록 입력 × GRU·LSTM)의 최선 대비 격차와 동률 여부를 본다.

# %%
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "engine").is_dir() and (p / "test" / "models").is_dir():
            return p
    raise RuntimeError(f"프로젝트 루트(engine/·test/models/)를 찾지 못했다: {start}")


ROOT = _project_root(Path(__file__).resolve())
TAG = "27e_regime_input_control_20261009"
STEM = "27e_regime_input_control"
RES = ROOT / "test" / "results" / TAG
IMG = ROOT / "test" / "images" / TAG


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D27 = _load("d27_for27e", ROOT / "test" / "models" / "27d_regime_seqpar_test.py")
X, M, B26 = D27.X, D27.M, D27.B26
R27C = ROOT / "test" / "results" / "27c_rnn_block_input_20261008"
R27D = D27.RES

SEEDS = D27.SEEDS
ALPHA = D27.ALPHA
SCOPES = D27.SCOPES
BLOCK = ("GRU-block", "LSTM-block")
RNN15 = ("GRU", "LSTM")
COMPARISONS = (  # (이름, A, B). 손실차 = A − B, 음수면 A가 낫다
    ("블록 입력 순환망 − 병렬(딥러닝)", BLOCK, D27.PAR_DEEP),
    ("블록 입력 순환망 − 병렬(특징 기반)", BLOCK, D27.PAR_FEAT),
    ("15분봉 구성 − 블록 구성 순환망", RNN15, BLOCK),
)
A_LABEL = {COMPARISONS[0][0]: ("블록 입력 순환망", "병렬 딥러닝"), COMPARISONS[1][0]: ("블록 입력 순환망", "병렬 특징 기반"),
           COMPARISONS[2][0]: ("15분봉 구성", "블록 구성")}
RNN4 = ("LSTM", "GRU", "LSTM-block", "GRU-block")
RNN4_LABEL = {"LSTM": "LSTM(15분봉 입력, 27d)", "GRU": "GRU(15분봉 입력, 27d)",
              "LSTM-block": "LSTM(블록 입력, 27c)", "GRU-block": "GRU(블록 입력, 27c)"}
SEQ_C, FEAT_C, DEEP_C = D27.GROUP_COLOR["순차"], D27.GROUP_COLOR["병렬(특징 기반)"], D27.GROUP_COLOR["병렬(딥러닝)"]
INK, MUTED, GRID = D27.INK, D27.MUTED, D27.GRID
_LINES: list[str] = []


def emit(t: str = "") -> None:
    _LINES.append(t)


fp = D27.fp
sname = lambda m: {"GRU-block": "GRU(블록)", "LSTM-block": "LSTM(블록)"}.get(m, D27.sname(m))


# %% [markdown]
# ## 입력과 게이트

# %%
def load_block(base: dict) -> dict:
    """27c 블록 입력 GRU·LSTM 예측(시드 → (종목, 구간) → 모델 → 예측). 시각·실제값이 27번과 같아야 한다."""
    miss = [s for s in SEEDS if not (R27C / f"27c_rnn_block_input_seed{s}_test_predictions.npz").exists()]
    if miss:
        raise RuntimeError(f"[시드 완전성 게이트: 27e] 27c 시드 예측 파일 없음: {miss}")
    out = {}
    for s in SEEDS:
        z = np.load(R27C / f"27c_rnn_block_input_seed{s}_test_predictions.npz")
        out[s] = {}
        for key, S in base.items():
            b = f"{key[0]}|{key[1]}|"
            for k in ("시각", "실제", *BLOCK):
                if b + k not in z:
                    raise RuntimeError(f"[27e 게이트] 시드 {s} {key}에 {k} 없음")
            if not np.array_equal(z[b + "시각"].astype("datetime64[ns]"), np.asarray(S["T"]).astype("datetime64[ns]")):
                raise RuntimeError(f"[27e 게이트] 시드 {s} {key} 평가 시각이 27번과 다르다")
            if not np.allclose(z[b + "실제"], S["act"], rtol=0, atol=1e-7):
                raise RuntimeError(f"[27e 게이트] 시드 {s} {key} 실제값이 27번과 다르다")
            out[s][key] = {m: z[b + m].astype(float) for m in BLOCK}
            for m in BLOCK:
                if not np.all(np.isfinite(out[s][key][m])) or np.any(out[s][key][m] <= 0):
                    raise RuntimeError(f"[27e 게이트] 시드 {s} {key} {m} 예측이 유한한 양수가 아니다")
    print(f"[입력] 27c 블록 입력 순환망 시드 {len(SEEDS)}개 × 칸 {len(base)}개 대조 통과", flush=True)
    return out


def build_cache(base: dict, seeds: dict, blk: dict) -> dict:
    """구간마다 27d와 같은 손실(시드별·시드 평균)과 국면에 블록 입력 모델을 더한다."""
    cache = {}
    for H in M.HORIZONS_H:
        Ls, Lm, Qf = D27.losses_by_seed(base, seeds, H)
        for s in SEEDS:
            for m in BLOCK:
                cols = {}
                for key in (k for k in base if k[1] == H):
                    S = base[key]
                    T = pd.DatetimeIndex(np.asarray(S["T"]).astype("datetime64[ns]"))
                    cols[key[0]] = pd.Series(M.qlike_vec(S["act"].astype(float), blk[s][key][m]), index=T)
                Ls[s][m] = pd.DataFrame(cols).sort_index()
        for m in BLOCK:
            Lm[m] = sum(Ls[s][m] for s in SEEDS) / len(SEEDS)
        cache[H] = (Ls, Lm, Qf)
    return cache


def reproduce_gate(cache: dict) -> float:
    """27d의 '순환 신경망 − 병렬(딥러닝)' 평균차를 같은 손실·국면으로 재현해야 이후 결과를 만든다."""
    ref = pd.read_csv(R27D / "27d_regime_seqpar_seqpar_dm.csv")
    ref = ref[ref["비교"] == "순환 신경망 − 병렬(딥러닝)"]
    worst = 0.0
    for H in M.HORIZONS_H:
        _, Lm, Qf = cache[H]
        for scope in SCOPES:
            mask = D27.scope_mask(Qf, scope)
            d = (D27.group_frame(Lm, RNN15) - D27.group_frame(Lm, D27.PAR_DEEP)).where(mask).mean(axis=1).dropna().to_numpy()
            r = ref[(ref["H"] == H) & (ref["국면"] == scope)]["평균차"].iloc[0]
            worst = max(worst, abs(float(d.mean()) - float(r)))
    if worst > 1e-12:
        raise RuntimeError(f"[재현 게이트] 27d 순환 신경망 − 병렬(딥러닝) 평균차를 재현하지 못했다(최대차 {worst:.3g})")
    print(f"[재현 게이트] 27d 묶음 평균차 재현(최대차 {worst:.1g})", flush=True)
    return worst


# %% [markdown]
# ## 묶음 검정(27d 3절과 같은 정의)

# %%
def group_tests(cache: dict) -> pd.DataFrame:
    rows = []
    for name, A, Bm in COMPARISONS:
        for H in M.HORIZONS_H:
            Ls, Lm, Qf = cache[H]
            for scope in SCOPES:
                mask = D27.scope_mask(Qf, scope)
                d = (D27.group_frame(Lm, A) - D27.group_frame(Lm, Bm)).where(mask).mean(axis=1).dropna().to_numpy()
                mu, se, p = B26.hac_mean_test(d)
                r = {"비교": name, "H": H, "국면": scope, "평균차": mu, "표준오차": se, "p": p, "시각수": len(d),
                     "A구성원수": len([m for m in A if m in Lm]), "B구성원수": len([m for m in Bm if m in Lm])}
                for s in SEEDS:
                    ds = (D27.group_frame(Ls[s], A) - D27.group_frame(Ls[s], Bm)).where(mask).mean(axis=1).dropna().to_numpy()
                    r[f"시드{s}_평균차"], _, r[f"시드{s}_p"] = B26.hac_mean_test(ds)
                rows.append(r)
    t = pd.DataFrame(rows)
    out = []
    for name, g in t.groupby("비교", sort=False):     # Holm은 비교마다 30칸(5구간 × 6범위)
        g = g.copy()
        g["p_holm"] = B26.holm(g["p"].fillna(1).to_numpy())
        same, opp = np.zeros(len(g), int), np.zeros(len(g), int)
        for s in SEEDS:
            sig = B26.holm(g[f"시드{s}_p"].fillna(1).to_numpy()) < ALPHA
            sg = np.sign(g[f"시드{s}_평균차"].to_numpy()) == np.sign(g["평균차"].to_numpy())
            same += sig & sg
            opp += sig & ~sg
        g["같은방향유의_시드수"], g["반대방향유의_시드수"] = same, opp
        a, b = A_LABEL[name]
        g["판정"] = np.where(g["p_holm"] >= ALPHA, "구분 안 됨", np.where(g["평균차"] < 0, f"{a} 우세", f"{b} 우세"))
        out.append(g)
    t = pd.concat(out, ignore_index=True)
    t.to_csv(RES / f"{STEM}_group_dm.csv", index=False)
    return t


# %% [markdown]
# ## 국면 최선 재검정(블록 입력 GRU·LSTM을 후보에 더함)

# %%
def regime_with_block(cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반환: (시드 평균 기준 칸별 순환망 4종의 순위·격차·동률, 시드별 동률 여부)."""
    rows, seed_rows = [], []
    for H in M.HORIZONS_H:
        Ls, Lm, Qf = cache[H]
        for scope in SCOPES:
            mask = D27.scope_mask(Qf, scope)
            t = D27.best_and_ties(Lm, mask).set_index("모델")
            mean_l = {m: v for m, v in D27.model_means(Lm, mask).items() if m != "naive"}   # 최선 선정·검정과 같은 시각 동일 가중
            order = sorted(mean_l, key=mean_l.get)
            for m in RNN4:
                rows.append({"H": H, "국면": scope, "모델": m, "최선": t["최선"].iloc[0], "순위": order.index(m) + 1,
                             "후보수": len(order), "격차": t.loc[m, "격차"], "p_holm": t.loc[m, "p_holm"],
                             "통계적동률": bool(t.loc[m, "통계적동률"])})
            for s in SEEDS:
                ts = D27.best_and_ties(Ls[s], mask).set_index("모델")
                for m in RNN4:
                    seed_rows.append({"H": H, "국면": scope, "시드": s, "모델": m, "최선": ts["최선"].iloc[0],
                                      "통계적동률": bool(ts.loc[m, "통계적동률"])})
        print(f"  [국면 재검정] {M.hlabel(H)} 완료", flush=True)
    rk = pd.DataFrame(rows)
    sd = pd.DataFrame(seed_rows)
    rk.to_csv(RES / f"{STEM}_rnn_rank_vs_best.csv", index=False)
    sd.to_csv(RES / f"{STEM}_rnn_ties_by_seed.csv", index=False)
    return rk, sd


def tie_seeds(sd: pd.DataFrame, H: int, scope: str, m: str) -> int:
    return int(sd[(sd["H"] == H) & (sd["국면"] == scope) & (sd["모델"] == m)]["통계적동률"].sum())


# %% [markdown]
# ## EDA 근거: 국면별 평균 회귀와 창 안의 최근 경로

# %%
H_COLOR = dict(zip(M.HORIZONS_H, D27.H_LINE_COLORS))   # 27d 선 그래프와 같은 구간 색(순차 파랑·병렬 주황과 겹치지 않음)


def eda_job(arg: tuple) -> list[dict]:
    """한 종목: 평가 시점마다 y = log(다음 H분 RV / 직전 H분 RV),
    z = 직전 창 RV² 중 뒤쪽 절반의 몫(0~1, 한쪽 절반이 가격 정지인 창도 포함),
    민감도 z2 = log(뒤쪽 절반 RV / 앞쪽 절반 RV)(양쪽 절반 모두 거래한 창만)."""
    from scipy.stats import spearmanr
    tk, cells = arg
    D = M.build_data(tk)
    grid = pd.DatetimeIndex(D["grid"])
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(D["r"], nan=0.0) ** 2)]
    rows = []
    for H, (T, act, nai, q) in cells.items():
        m = H // 15
        j = grid.get_indexer(pd.DatetimeIndex(T))
        if np.any(j < m):
            raise RuntimeError(f"[EDA 게이트] {tk} {H}: 평가 시각을 15분 격자에서 찾지 못했다")
        rec = np.sqrt(np.maximum(cs[j] - cs[j - m], 0.0))
        if not np.allclose(rec, nai, rtol=1e-4, atol=1e-7):
            raise RuntimeError(f"[EDA 게이트] {tk} {H}: 격자에서 다시 만든 직전 RV가 저장된 naive 입력과 다르다")
        ok = (act > 0) & (nai > 0)
        y = np.full(len(act), np.nan)
        y[ok] = np.log(act[ok] / nai[ok])
        if m >= 2:
            h = m // 2
            a1, a2 = np.maximum(cs[j - h] - cs[j - m], 0.0), np.maximum(cs[j] - cs[j - h], 0.0)
            tot = a1 + a2
            z = np.full(len(act), np.nan)
            z[tot > 0] = a2[tot > 0] / tot[tot > 0]
            okz2 = (a1 > 0) & (a2 > 0)
            z2 = np.full(len(act), np.nan)
            z2[okz2] = 0.5 * np.log(a2[okz2] / a1[okz2])
        else:
            z = np.full(len(act), np.nan)
            z2 = np.full(len(act), np.nan)
        for k in range(5):
            sel = (q == k) & np.isfinite(y)
            selz, selz2 = sel & np.isfinite(z), sel & np.isfinite(z2)
            rho = float(spearmanr(y[selz], z[selz])[0]) if selz.sum() >= 30 else np.nan
            rho2 = float(spearmanr(y[selz2], z2[selz2])[0]) if selz2.sum() >= 30 else np.nan
            rows.append({"종목": tk, "H": H, "국면": f"Q{k + 1}", "n": int(sel.sum()), "n_경로": int(selz.sum()),
                         "n_양쪽거래": int(selz2.sum()), "평균_log비": float(np.mean(y[sel])) if sel.any() else np.nan,
                         "rho_경로": rho, "rho_양쪽거래": rho2})
    return rows


def eda_regime(base: dict, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import binomtest, wilcoxon
    jobs = {}
    for (tk, H), S in base.items():
        q = np.digitize(S["nai"], np.quantile(S["nai_tr"], [.2, .4, .6, .8]))
        jobs.setdefault(tk, {})[H] = (np.asarray(S["T"]).astype("datetime64[ns]"), S["act"].astype(float),
                                      S["nai"].astype(float), q)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        per = pd.DataFrame([r for rs in ex.map(eda_job, sorted(jobs.items())) for r in rs])
    per.to_csv(RES / f"{STEM}_eda_per_ticker.csv", index=False)
    rows = []
    for H in M.HORIZONS_H:
        for k in range(1, 6):
            g = per[(per["H"] == H) & (per["국면"] == f"Q{k}")]
            rr = g["rho_경로"].dropna()
            r2 = g["rho_양쪽거래"].dropna()
            npos = int((rr > 0).sum())
            p = binomtest(npos, len(rr), 0.5).pvalue if len(rr) else np.nan
            gm = g["평균_log비"].dropna()
            use2 = (g["n_양쪽거래"] / g["n"].where(g["n"] > 0)).dropna()
            rows.append({"H": H, "국면": f"Q{k}", "종목수": len(rr), "rho_중앙값": rr.median() if len(rr) else np.nan,
                         "rho_양수종목": npos, "부호검정_p": p, "평균_log비_종목수": len(gm),
                         "평균_log비_중앙값": gm.median() if len(gm) else np.nan, "평균_log비_음수종목": int((gm < 0).sum()),
                         "민감도_rho_중앙값": r2.median() if len(r2) else np.nan, "민감도_종목수": len(r2),
                         "민감도_사용비율_중앙값": use2.median() if len(use2) else np.nan})
    t = pd.DataFrame(rows)
    ok = t["부호검정_p"].notna()
    t.loc[ok, "부호검정_p_holm"] = B26.holm(t.loc[ok, "부호검정_p"].to_numpy())
    cmp_rows = []
    for H in M.HORIZONS_H:
        a = per[(per["H"] == H) & (per["국면"] == "Q5")].set_index("종목")["rho_경로"]
        b = per[(per["H"] == H) & (per["국면"] == "Q1")].set_index("종목")["rho_경로"]
        d = (a - b).dropna()
        p = float(wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else np.nan
        cmp_rows.append({"H": H, "종목수": len(d), "차_중앙값(Q5−Q1)": d.median() if len(d) else np.nan,
                         "Q5가_큰_종목": int((d > 0).sum()), "윌콕슨_p": p})
    c = pd.DataFrame(cmp_rows)
    okc = c["윌콕슨_p"].notna()
    c.loc[okc, "윌콕슨_p_holm"] = B26.holm(c.loc[okc, "윌콕슨_p"].to_numpy())
    t.to_csv(RES / f"{STEM}_eda_regime.csv", index=False)
    c.to_csv(RES / f"{STEM}_eda_q5_vs_q1.csv", index=False)
    print("[EDA] 국면별 평균 회귀·창 안 경로 완료", flush=True)
    return t, c


def fig_eda(t: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.3))
    x = np.arange(1, 6)
    for H in M.HORIZONS_H:
        g = t[t["H"] == H].sort_values("국면")
        axes[0].plot(x, g["평균_log비_중앙값"], color=H_COLOR[H], marker="o", lw=2.2, label=M.hlabel(H))
        if g["rho_중앙값"].notna().any():
            axes[1].plot(x, g["rho_중앙값"], color=H_COLOR[H], marker="o", lw=2.2, label=M.hlabel(H))
    axes[0].axhline(0, color=MUTED, lw=1)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[0].set_title("(왼쪽) 다음 변동성 ÷ 직전 변동성(로그, 종목 중앙값)\n0보다 아래 = 다음 변동성이 직전보다 줄어든다(평균 회귀)", fontsize=12.5, loc="left")
    axes[1].set_title("(오른쪽) 직전 구간 뒤쪽 절반에 변동성이 몰린 정도와 왼쪽 값의 순위 상관\n"
                      "(Spearman, 종목 중앙값, 0보다 위 = 뒤쪽에 몰렸을수록 덜 줄어든다, 15분은 해당 없음)", fontsize=12.5, loc="left")
    for ax in axes:
        ax.set_xticks(x, ["Q1\n잔잔", "Q2", "Q3", "Q4", "Q5\n요동"])
        ax.set_xlabel("직전 변동성 국면(종목별 학습 구간 5분위)")
        ax.grid(axis="y", color=GRID)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.legend(frameon=False, title="예측 구간", fontsize=10)
    fig.tight_layout()  # 제목은 보고서·포스터 캡션이 맡는다
    p = IMG / f"{STEM}_fig3_eda_recent_path.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# %% [markdown]
# ## 그림

# %%
def _grid_axes(ax, ylabels) -> None:
    Hs = list(M.HORIZONS_H)
    ax.set_xlim(0, len(Hs))
    ax.set_ylim(len(SCOPES), 0)
    ax.set_xticks(np.arange(len(Hs)) + 0.5, [M.hlabel(H) for H in Hs], fontsize=10.5)
    ax.set_yticks(np.arange(len(SCOPES)) + 0.5, ylabels, fontsize=10.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(1, color=INK, lw=1.0)


def fig_group(t: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    Hs = list(M.HORIZONS_H)
    for k, (ax, (name, _, _)) in enumerate(zip(axes, COMPARISONS)):
        a, b = A_LABEL[name]
        ca, cb = (SEQ_C, DEEP_C) if k == 0 else (SEQ_C, FEAT_C) if k == 1 else ("#5b3fa0", SEQ_C)
        col = {f"{a} 우세": ca, f"{b} 우세": cb, "구분 안 됨": GRID}
        g = t[t["비교"] == name]
        for i, scope in enumerate(SCOPES):
            for j, H in enumerate(Hs):
                x = g[(g["H"] == H) & (g["국면"] == scope)].iloc[0]
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=col[x["판정"]],
                                           alpha=0.8 if x["판정"] != "구분 안 됨" else 1.0, edgecolor="white", lw=2))
                tc = "white" if x["판정"] != "구분 안 됨" else INK
                ax.text(j + 0.5, i + 0.42, f"{x['평균차']:+.3f}", ha="center", va="center", fontsize=10.5, color=tc, fontweight="bold")
                ax.text(j + 0.5, i + 0.72, f"시드 {int(x['같은방향유의_시드수'])}/5", ha="center", va="center", fontsize=8.5, color=tc)
        _grid_axes(ax, ["전체 기간", "Q1 잔잔", "Q2", "Q3", "Q4", "Q5 요동"] if k == 0 else [""] * len(SCOPES))
        ax.set_title(f"{name}\n(음수 = {a} 쪽이 낫다)", fontsize=12, loc="left", color=INK)
        ax.legend(handles=[Patch(facecolor=col[c], label=c) for c in col], loc="upper center",
                  bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("입력을 맞춘 묶음 검정: 블록 입력 순환망(27c, 병렬 딥러닝과 같은 입력) 대 병렬 묶음(시드 5개 평균, Holm p<0.05)\n"
                 "칸 아래 숫자 = 시드별로 다시 검정해 같은 방향으로 유의했던 시드 수", fontsize=13.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = IMG / f"{STEM}_fig1_group_dm.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_rnn_rank(rk: pd.DataFrame, sd: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
    Hs = list(M.HORIZONS_H)
    TIE_C, LOSE_C = "#cfe0f6", "#f3f2ee"
    for k, (ax, m) in enumerate(zip(axes.ravel(), RNN4)):
        g = rk[rk["모델"] == m]
        for i, scope in enumerate(SCOPES):
            for j, H in enumerate(Hs):
                x = g[(g["H"] == H) & (g["국면"] == scope)].iloc[0]
                v = float(x["격차"])
                is_best = x["모델"] == x["최선"]
                fc = SEQ_C if is_best else (TIE_C if x["통계적동률"] else LOSE_C)
                tc = "white" if is_best else INK
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=fc, edgecolor="white", lw=2))
                head = "최선" if is_best else f"{int(x['순위'])}위 · {v:+.3f}"
                ax.text(j + 0.5, i + 0.38, head, ha="center", va="center", fontsize=10.5, fontweight="bold", color=tc)
                ax.text(j + 0.5, i + 0.68, ("최선" if is_best else "동률" if x["통계적동률"] else "열세")
                        + f" · 시드 동률 {tie_seeds(sd, H, scope, m)}/5", ha="center", va="center", fontsize=8.5, color=tc)
        _grid_axes(ax, ["전체 기간", "Q1 잔잔", "Q2", "Q3", "Q4", "Q5 요동"] if k % 2 == 0 else [""] * len(SCOPES))
        ax.set_title(RNN4_LABEL[m], fontsize=12.5, loc="left", color=SEQ_C, fontweight="bold")
    fig.legend(handles=[Patch(facecolor=SEQ_C, label="칸 최선"), Patch(facecolor=TIE_C, label="최선과 통계적 동률(Holm p ≥ 0.05)"),
                        Patch(facecolor=LOSE_C, edgecolor=GRID, label="최선보다 유의하게 나쁨(열세)")],
               loc="lower center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.005))
    n_c = f"{int(rk['후보수'].max())}개(4·12시간은 TTM 제외 {int(rk['후보수'].min())}개)"
    fig.suptitle(f"순환 신경망 4종의 국면별 위치: 후보 {n_c}(27d 모델 + 블록 입력 GRU·LSTM) 중 순위와 최선 대비 QLIKE 격차(시드 5개 평균)\n"
                 "시드 동률 = 시드마다 최선을 다시 골라 검정했을 때 그 모델이 최선이거나 동률이었던 시드 수", fontsize=13, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    p = IMG / f"{STEM}_fig2_rnn_rank.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


# %% [markdown]
# ## 보고서

# %%
def write_report(t: pd.DataFrame, rk: pd.DataFrame, sd: pd.DataFrame, figs: dict, worst: float,
                 et: pd.DataFrame, ec: pd.DataFrame, sd_tickers: list[str]) -> None:
    rel = lambda p: os.path.relpath(p, RES)
    emit("# 27e번: 변동성 국면별 결과의 입력 통제 점검(시드 5개)")
    emit()
    emit(D27.standard_header(sorted({tk for tk in sd_tickers}), D27.BASE_TRANSFORMS + [
        ("블록 입력 GRU·LSTM(27c 예측 재사용)", "병렬 딥러닝과 같은 H분 블록 로그 RV 입력·겹치지 않는 블록 표본으로 학습한 순환망. 재학습 없음"),
        ("국면별 EDA: 다음/직전 RV 로그비, 직전 창 뒤쪽 절반 몫", "요동 국면에서 창 안 경로 정보가 중요한지 보는 기술 통계(3절)")]))
    emit()
    emit("27번·27c의 저장된 평가 예측(시드 0~4)만 읽고 재학습하지 않았다.")
    emit()
    emit("## 0. 이 회차가 한 것과 이유")
    emit()
    emit("27d는 요동 국면(Q5)의 1시간·4시간·12시간 최선이 순환 신경망이고, 순환 신경망 묶음이 병렬 딥러닝 묶음보다 요동 국면에서 유의하게 낫다고 "
         "보고했다. 그러나 27d의 GRU·LSTM은 **15분봉 96개의 (수익률, 절대수익률)** 을 읽고 15분마다 겹치는 표본으로 학습했으며, 병렬 딥러닝은 "
         "**H분 블록 로그 RV 이력**을 겹치지 않는 블록 표본으로 학습했다. 27c는 GRU·LSTM에 병렬 딥러닝과 같은 블록 입력·표본을 주어 이 차이를 "
         "전체 기간에서만 통제했다. 이 회차는 27c의 블록 입력 GRU·LSTM을 27d와 같은 국면(직전 H분 RV의 종목별 학습 구간 5분위)으로 나눠, "
         "27d의 국면별 결론이 입력 차이 때문인지 확인한다.")
    emit()
    emit("- **손실·국면·검정**: 27d와 같다(QLIKE, 시드별 손실의 평균, 직전 RV 학습 5분위, DM·Newey-West HAC·Holm).")
    emit(f"- **게이트**: 27c 시드 0~4 예측 파일이 모두 있고, 100칸 모두 평가 시각·실제값이 27번과 같으며, 27d의 '순환 신경망 − 병렬(딥러닝)' "
         f"묶음 평균차를 같은 코드 경로로 재현해야(최대차 {worst:.1g}) 결과를 만든다. 이번 실행은 모두 통과했다.")
    emit("- **남는 한계**: 블록 입력 순환망과 병렬 딥러닝도 학습 절차(학습률 탐색·검증 구간·재적합)는 맞추지 않았다(27c와 같음). "
         "iTransformer·S-Mamba는 블록 수익률 변수가 하나 더 있다.")
    emit()

    # 1. 묶음 검정
    emit("## 1. 입력을 맞춘 묶음 검정")
    emit()
    emit(f"![입력을 맞춘 묶음 검정]({rel(figs['group'])})")
    emit()
    emit("**읽는 법**: 세 비교마다 행은 범위(전체 기간, 국면 Q1~Q5), 열은 예측 구간이다. 숫자는 묶음 평균 QLIKE 차(A − B)로 음수면 A가 낫다. "
         "색칠한 칸은 Holm 보정 후 유의(p < 0.05), 회색은 구분 안 됨이다. \"시드 k/5\"는 시드마다 다시 검정해 같은 방향으로 유의했던 시드 수다.")
    emit()
    for name, A, Bm in COMPARISONS:
        g = t[t["비교"] == name]
        a, b = A_LABEL[name]
        emit(f"### {name}")
        emit()
        emit(f"비교 집단: A = {', '.join(sname(x) for x in A)} / B = {', '.join(sname(x) for x in Bm) if len(Bm) <= 7 else f'{len(Bm)}종(TTM은 4·12시간 평가 제외라 그 구간은 14종)'}. "
             "H0: 두 묶음 평균의 기대 QLIKE가 같다. H1: 같지 않다(양측). 비교마다 30칸에 Holm 보정.")
        emit()
        emit("| 범위 | " + " | ".join(M.hlabel(H) for H in M.HORIZONS_H) + " |")
        emit("| :--- | " + " | ".join([":---"] * len(M.HORIZONS_H)) + " |")
        for scope in SCOPES:
            cells = []
            for H in M.HORIZONS_H:
                x = g[(g["H"] == H) & (g["국면"] == scope)].iloc[0]
                cells.append(f"{x['판정']} {x['평균차']:+.3f} (p={fp(x['p_holm'])}, 시드 {int(x['같은방향유의_시드수'])}/5"
                             + (f", 반대 {int(x['반대방향유의_시드수'])}" if x["반대방향유의_시드수"] else "") + ")")
            emit(f"| {scope if scope != '전체' else '전체 기간'} | " + " | ".join(cells) + " |")
        emit()
        cnt = g["판정"].value_counts()
        q5 = g[g["국면"] == "Q5"]
        emit(f"**해석**: 30칸 중 {a} 우세 {int(cnt.get(f'{a} 우세', 0))}칸, {b} 우세 {int(cnt.get(f'{b} 우세', 0))}칸, 구분 안 됨 "
             f"{int(cnt.get('구분 안 됨', 0))}칸이다. 요동 국면(Q5) 다섯 구간의 판정은 "
             + ", ".join(f"{M.hlabel(int(r.H))} {r['판정']}({r['평균차']:+.3f})" for _, r in q5.iterrows()) + "이다. "
             "유의 칸은 H0를 기각하고 해당 쪽 묶음의 기대 손실이 작다는 뜻이며, 구분 안 됨은 차이의 증거가 부족하다는 뜻이다(같다는 증명은 아니다).")
        emit()

    # 2. 국면 최선 재검정
    emit("## 2. 순환 신경망 4종의 국면별 위치(블록 입력 GRU·LSTM을 후보에 더해 국면 최선 재검정)")
    emit()
    emit(f"![순환 신경망 국면별 위치]({rel(figs['rank'])})")
    emit()
    n_c = f"{int(rk['후보수'].max())}개(4·12시간은 TTM 제외 {int(rk['후보수'].min())}개)"
    emit(f"**읽는 법**: 27d의 후보에 블록 입력 GRU·LSTM을 더한 {n_c}(naive 제외)로 칸마다 최선을 다시 고르고, 최선 대 나머지 각 모델의 DM 검정에 "
         "Holm 보정을 건다(27d 2절과 같은 정의, 후보가 늘어 Holm 보정 대상이 27d보다 2개 많다). H0: 최선과 기대 QLIKE가 같다. 칸 글씨는 순위와 최선 대비 "
         "격차, 진한 파랑은 칸 최선, 연한 파랑은 최선과 통계적 동률, 회색은 최선보다 유의하게 나쁜(열세) 칸이다. 최선·순위·격차·검정은 모두 같은 "
         "가중(시각마다 그 국면에 든 종목 평균을 낸 뒤 시각 평균)으로 계산한다.")
    emit()
    changed = []
    for H in M.HORIZONS_H:
        for scope in SCOPES:
            b_new = rk[(rk["H"] == H) & (rk["국면"] == scope)]["최선"].iloc[0]
            b_old = pd.read_csv(R27D / "27d_regime_seqpar_regime_best_ties.csv").query("H == @H and 국면 == @scope")["최선"].iloc[0]
            if b_new != b_old:
                changed.append(f"{M.hlabel(H)} {scope}: {b_old} → {b_new}")
    emit(f"- **최선이 바뀐 칸**: {'; '.join(changed) if changed else '해당 없음(블록 입력 순환망이 최선이 된 칸 없음, 30칸 모두 27d와 같은 최선)'}.")
    emit()
    emit("| 예측 구간 | 국면 | 최선 | " + " | ".join(RNN4_LABEL[m] for m in RNN4) + " |")
    emit("| :--- | :--- | :--- | " + " | ".join([":---"] * len(RNN4)) + " |")
    for H in M.HORIZONS_H:
        for scope in SCOPES:
            g = rk[(rk["H"] == H) & (rk["국면"] == scope)].set_index("모델")
            cells = []
            for m in RNN4:
                x = g.loc[m]
                st_ = "최선" if m == x["최선"] else ("동률" if x["통계적동률"] else "열세")
                cells.append(f"{int(x['순위'])}위 {st_} {x['격차']:+.3f} (p={fp(x['p_holm'])}, 시드 동률 {tie_seeds(sd, H, scope, m)}/5)")
            emit(f"| {M.hlabel(H)} | {scope if scope != '전체' else '전체 기간'} | {g['최선'].iloc[0]} | " + " | ".join(cells) + " |")
    emit()
    q5 = rk[rk["국면"] == "Q5"]
    blk_q5 = q5[q5["모델"].isin(BLOCK)]
    rnn_q5 = q5[q5["모델"].isin(RNN15)]
    blk_tie = sorted({M.hlabel(int(H)) for H, g in blk_q5.groupby("H") if g["통계적동률"].any()}, key=lambda s: list(map(M.hlabel, M.HORIZONS_H)).index(s))
    rnn_tie = sorted({M.hlabel(int(H)) for H, g in rnn_q5.groupby("H") if g["통계적동률"].any()}, key=lambda s: list(map(M.hlabel, M.HORIZONS_H)).index(s))
    emit(f"**해석(요동 국면 Q5)**: 15분봉 입력 GRU·LSTM 중 하나가 최선이거나 최선과 동률인 구간은 {', '.join(rnn_tie) or '해당 없음(없음)'}이고, "
         f"블록 입력 GRU·LSTM 중 하나가 최선과 동률인 구간은 {', '.join(blk_tie) or '해당 없음(모든 구간에서 열세)'}이다. "
         "블록 입력 순환망의 Q5 순위는 " + ", ".join(f"{M.hlabel(int(H))} {int(g['순위'].min())}위" for H, g in blk_q5.groupby("H")) + "다.")
    emit()
    emit("## 3. EDA 근거: 요동 국면에서 직전 창 안의 최근 경로가 더 중요한가")
    emit()
    emit(f"![국면별 평균 회귀와 최근 경로]({rel(figs['eda'])})")
    emit()
    emit("**무엇을 보는가**: 2절에서 요동 국면 1시간~12시간 최선은 15분봉을 차례로 읽는 순환 신경망이었고, 같은 순환망이라도 H분 블록 RV만 읽으면 순위가 "
         "내려갔다. 그렇다면 요동 국면에서는 직전 H분을 한 숫자(블록 RV)로 요약할 때 잃는 정보, 곧 창 안에서 변동성이 어떻게 움직였는지가 다음 변동성과 "
         "관련이 있어야 한다. 평가 구간의 예측 시점마다 두 값을 만든다. y = log(다음 H분 RV ÷ 직전 H분 RV)는 다음 변동성이 직전 수준에서 얼마나 바뀌는지, "
         "z = 직전 창 RV² 중 뒤쪽 절반이 차지하는 몫(0~1)은 직전 창 안에서 변동성이 뒤쪽에 몰렸는지(커지는 중) 앞쪽에 몰렸는지(잦아드는 중)다. z는 블록 "
         "RV 하나로는 알 수 없고 15분봉 경로를 읽어야 알 수 있다. 한쪽 절반이 가격 정지(0)인 창도 z가 정의되므로 국면마다 표본이 따로 골라지지 않는다. "
         "민감도로 양쪽 절반 모두 거래한 창만 쓴 로그비 정의도 함께 적는다. 15분은 직전 창이 15분봉 1개라 z를 만들 수 없다(해당 없음). "
         "게이트: 15분 격자에서 다시 만든 직전 RV가 저장된 naive 입력과 모든 시점에서 같아야 한다(통과).")
    emit()
    emit("**검정**: 비교 집단은 종목(최대 20개)이다. 종목마다 국면별 Spearman 순위 상관 ρ(y, z)를 구한다. (1) 국면별 부호 검정 — H0: ρ가 양수인 종목과 "
         "음수인 종목이 반반이다(창 안 경로가 다음 변동성 변화와 관계없다), H1: 반반이 아니다. 구간 × 국면 20칸에 Holm 보정. (2) Q5 대 Q1 — 같은 종목의 "
         "ρ(Q5) − ρ(Q1)에 윌콕슨 부호순위 검정, H0: 차이의 중앙값이 0이다, H1: 0이 아니다. 구간 4개에 Holm 보정. 시점 간 자기상관 때문에 시점 대신 종목을 "
         "단위로 썼다. 다만 종목들은 같은 기간의 같은 시장 충격을 겪어 완전히 독립이 아니므로 p값은 근사이고 다소 낙관적일 수 있다.")
    emit()
    emit("| 예측 구간 | 국면 | 평균 회귀: log 비 중앙값(음수 종목/유효 종목) | 경로 상관 ρ 중앙값(양수 종목/유효 종목) | 부호 검정 Holm p | 민감도 ρ 중앙값(양쪽 거래 창, 종목 수, 사용 비율) |")
    emit("| :--- | :--- | ---: | ---: | ---: | ---: |")
    for _, r in et.iterrows():
        if np.isfinite(r["rho_중앙값"]):
            rho = f"{r['rho_중앙값']:+.3f}({int(r['rho_양수종목'])}/{int(r['종목수'])})"
            ph = fp(r["부호검정_p_holm"])
            sens = (f"{r['민감도_rho_중앙값']:+.3f}({int(r['민감도_종목수'])}종목, {r['민감도_사용비율_중앙값']:.0%})"
                    if np.isfinite(r["민감도_rho_중앙값"]) else "해당 없음(양쪽 거래 창 30개 미만)")
        else:
            rho = ph = sens = "해당 없음(15분은 창이 1봉)"
        lr = (f"{r['평균_log비_중앙값']:+.3f}({int(r['평균_log비_음수종목'])}/{int(r['평균_log비_종목수'])})"
              if np.isfinite(r["평균_log비_중앙값"]) else "해당 없음(표본 없음)")
        emit(f"| {M.hlabel(int(r['H']))} | {r['국면']} | {lr} | {rho} | {ph} | {sens} |")
    emit()
    emit("| 예측 구간 | ρ(Q5) − ρ(Q1) 중앙값 | Q5가 큰 종목 | 윌콕슨 Holm p |")
    emit("| :--- | ---: | ---: | ---: |")
    for _, r in ec.iterrows():
        if not np.isfinite(r["차_중앙값(Q5−Q1)"]):
            emit(f"| {M.hlabel(int(r['H']))} | 해당 없음(15분은 창이 1봉) | 해당 없음 | 해당 없음 |")
        else:
            emit(f"| {M.hlabel(int(r['H']))} | {r['차_중앙값(Q5−Q1)']:+.3f} | {int(r['Q5가_큰_종목'])}/{int(r['종목수'])} | {fp(r['윌콕슨_p_holm'])} |")
    emit()
    q1m = et[et["국면"] == "Q1"]["평균_log비_중앙값"].dropna()
    q5e = et[et["국면"] == "Q5"]
    ev = et[et["rho_중앙값"].notna()]
    q5r, q1r = ev[ev["국면"] == "Q5"]["rho_중앙값"], ev[ev["국면"] == "Q1"]["rho_중앙값"]
    top = [M.hlabel(int(H)) for H, g in ev.groupby("H") if g.loc[g["rho_중앙값"].idxmax(), "국면"] == "Q5"]
    sig_up = [M.hlabel(int(r['H'])) for _, r in ec.iterrows() if np.isfinite(r.get('윌콕슨_p_holm', np.nan)) and r['윌콕슨_p_holm'] < ALPHA and r['차_중앙값(Q5−Q1)'] > 0]
    not_sig = [M.hlabel(int(r['H'])) for _, r in ec.iterrows() if np.isfinite(r.get('윌콕슨_p_holm', np.nan)) and not (r['윌콕슨_p_holm'] < ALPHA)]
    emit(f"**해석**: (1) 평균 회귀: 잔잔한 국면(Q1)은 다음 변동성이 직전보다 커지고(log 비 중앙값 {q1m.min():+.2f}~{q1m.max():+.2f}), 요동 국면(Q5)은 "
         f"작아진다({q5e['평균_log비_중앙값'].min():+.2f}~{q5e['평균_log비_중앙값'].max():+.2f}). 국면을 직전 RV로 나눴으므로 이 방향은 평균으로의 회귀에서 "
         "예상되는 것이고, 모든 학습 모델이 배울 수 있는 성질이라 그 자체로 모델을 가르는 근거는 아니다. (2) 창 안 경로: ρ 중앙값은 30분~12시간 Q1에서 "
         f"{q1r.min():+.3f}~{q1r.max():+.3f}이고, Q5에서는 {q5r.min():+.3f}~{q5r.max():+.3f}로 약하지만 모든 구간에서 양(+)이다. ρ가 가장 큰 국면이 Q5인 구간은 "
         f"{', '.join(top) or '해당 없음(없음)'}이고(국면이 커질수록 단조롭게 커지지는 않는다), Q5가 Q1보다 유의하게 큰 구간은 {', '.join(sig_up) or '해당 없음(없음)'}, "
         f"구분 안 되는 구간은 {', '.join(not_sig) or '해당 없음(없음)'}이다. ρ가 양수라는 것은 직전 창 뒤쪽에 변동성이 몰려 있으면 다음 변동성이 덜 줄어든다는 "
         "뜻이고, 이 정보는 블록 RV 하나에는 없다.")
    emit()
    g_bf = t[(t["비교"] == COMPARISONS[1][0]) & (t["국면"] == "Q5")]
    emit("**알고리즘 특성과의 연결(해석이며 인과 식별은 아니다)**: 창 안 경로 정보는 15분봉을 받는 모델, 곧 15분봉 순환 신경망, 1·4·16봉 RV·지연 |r| 같은 "
         "다중 척도 특징을 쓰는 트리·커널, 15분 수익률을 재귀로 읽는 GARCH 3종이 받는다. H분 블록 RV 이력만 받는 병렬 딥러닝은 받지 못한다. 요동 국면 "
         "1시간~12시간에서 15분봉 순환 신경망이 최선이고 트리·커널이 그와 동률이며 병렬 딥러닝이 모두 열세라는 2절·27d 결과는 이 구분과 같은 방향이다. "
         "그러나 반례도 있다. GARCH 3종은 같은 정보를 받고도 요동 국면 4·12시간에서 모두 열세이고, 경로를 보지 못하는 블록 입력 순환망은 요동 국면에서 "
         "트리·커널 묶음과 구분되지 않는다(평균차 " + ", ".join(f"{M.hlabel(int(r.H))} {r['평균차']:+.3f}" for _, r in g_bf.iterrows())
         + ", 모두 구분 안 됨). 따라서 경로 정보는 요동 국면 결과를 설명하는 한 요인일 뿐이고, 모델의 비선형 표현력과 학습 절차도 함께 작용한다.")
    emit()
    emit("**뒷받침 문헌(방향이 비슷한 관찰이며, 문헌은 주가지수의 일·월 단위 RV라 15분~12시간 암호화폐로 옮기는 것은 유추다)**: Corsi(2009)의 HAR는 변동성이 "
         "여러 시간 척도의 성분으로 움직이고 짧은 척도 성분이 따로 예측력을 갖는다고 보고했다. 이 연구의 15분봉 경로·다중 척도 특징이 그 짧은 척도 성분에 "
         "해당한다. Bollerslev·Patton·Quaedvlieg(2016)의 HARQ는 RV의 측정오차가 큰 시기(대개 고변동 시기)에는 과거 RV의 지속성을 낮춰 써야 예측이 좋아진다고 "
         "보고했다. 요동 국면에서 다음 변동성이 직전보다 작아진다는 (1)과 비슷한 방향의 관찰이다. Bucci(2020)는 실현 변동성 예측에서 LSTM 등 순환 신경망이 "
         "HAR 등 계량 모형과 경쟁하거나 더 낫다고 보고했다. 요동 국면의 순환 신경망 결과와 같은 방향이다.")
    emit()
    emit("## 4. 결론: 27d 국면 결과를 어떻게 써야 하는가")
    emit()
    g0 = t[t["비교"] == COMPARISONS[0][0]]
    g2 = t[t["비교"] == COMPARISONS[2][0]]
    n_blk = int((g0["판정"] == "블록 입력 순환망 우세").sum())
    q5_blk = int(((g0["국면"] == "Q5") & (g0["판정"] == "블록 입력 순환망 우세")).sum())
    q5_in = g2[g2["국면"] == "Q5"]
    emit(f"1. **순환 신경망 묶음 > 병렬 딥러닝 묶음은 입력을 맞춰도 남는다**: 블록 입력 순환망이 병렬 딥러닝보다 30칸 중 {n_blk}칸에서 유의하게 낫고, "
         f"요동 국면 다섯 구간 중 {q5_blk}구간에서 유의하게 낫다. 따라서 27d의 이 묶음 결론은 입력 차이만으로 생긴 것이 아니다(학습 절차 차이는 남는다).")
    q5b = {int(H): g["최선"].iloc[0] for H, g in q5.groupby("H")}
    rnn_best = [H for H in M.HORIZONS_H if q5b[H] in RNN15]
    blk_lose = [M.hlabel(int(H)) for H, g in blk_q5.groupby("H") if int(H) in rnn_best and not g["통계적동률"].any()]
    emit("2. **요동 국면의 '최선' 자리는 15분봉 구성의 순환 신경망에만 해당한다**: 요동 국면에서 최선이 순환 신경망인 구간은 "
         + ", ".join(f"{M.hlabel(H)} {q5b[H]}" for H in rnn_best) + "이고 모두 15분봉 96개를 읽는 구성이다. 블록 구성(27c) GRU·LSTM은 위 2절처럼 "
         f"이 칸들에서 순위가 내려가고 {', '.join(blk_lose) or '해당 없음(없음)'}에서는 최선보다 유의하게 나쁘다. 두 구성의 직접 비교"
         "(15분봉 구성 − 블록 구성) 판정은 요동 국면에서 "
         + ", ".join(f"{M.hlabel(int(r.H))} {r['판정']}({r['평균차']:+.3f})" for _, r in q5_in.iterrows())
         + "다. 12시간은 직접 비교가 유의하지 않아 어느 구성이 나은지 증거가 부족하다. 두 구성은 입력 표현(15분봉 96개 대 H분 블록 96·96·96·60·30개, "
         "곧 긴 구간에서는 보는 과거 길이까지 다름), 학습 표본(15분마다 겹침 대 겹치지 않는 블록), 배치 크기가 함께 다르므로, 이 차이를 '입력 표현만의 "
         "효과'로 식별할 수는 없다. 쓸 수 있는 말은 \"요동 국면 최선은 15분봉을 차례로 읽는 구성의 순환 신경망이었다\"까지다.")
    emit("3. **포스터 문구**: \"요동 국면은 순환 신경망이 최선\"은 \"15분봉을 읽는 순환 신경망이 손실이 가장 작았다(트리·커널과는 통계적으로 구분되지 "
         "않음)\"로 쓰고, 순차 대 병렬 묶음 차이는 \"입력·표본을 맞춰도 순환 신경망 묶음이 병렬 딥러닝 묶음보다 낫다\"로 쓴다. 구조만의 효과나 "
         "입력 표현만의 효과라고 쓰지 않는다(학습 절차·표본 구성 미통제).")
    emit()
    emit("## 5. 한계")
    emit()
    emit("1. 블록 입력 순환망과 병렬 딥러닝은 입력·표본은 같지만 학습 절차와 변수 수(iTransformer·S-Mamba)는 다르다.")
    emit("2. 15분봉 구성과 블록 구성은 입력 표현·관측 길이·학습 표본·배치가 함께 다르다. 입력 표현만의 효과를 말하려면 관측 길이·학습 시점·최적화 조건을 맞춘 별도 비교가 필요하다.")
    emit("3. 반대 방향 통제(병렬 딥러닝에 15분봉 입력)는 하지 않았다(27c 0절의 이유). 따라서 병렬 딥러닝도 15분봉을 읽으면 요동 국면에서 나아지는지는 답하지 못한다.")
    emit("4. 국면별 표본은 12시간 국면당 관측 0.2만~0.3만으로 작다(27d 2절).")
    emit("5. 3절 EDA는 평가 구간 표본의 기술 통계이고, 알고리즘 결과와의 연결은 해석이다(모델이 실제로 그 정보를 쓰는지는 직접 측정하지 않았다).")
    (RES / f"{STEM}_report.md").write_text("\n".join(_LINES), encoding="utf-8")


# %%
def selftest() -> None:
    assert set(BLOCK).isdisjoint(D27.GROUP_OF) and set(RNN15) <= set(D27.SEQ)
    L = {"x": pd.DataFrame({"a": [1.0, 2.0]}), "y": pd.DataFrame({"a": [3.0, 4.0]})}
    assert np.allclose(D27.group_frame(L, ("x", "y")).to_numpy().ravel(), [2.0, 3.0])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args(argv)
    selftest()
    D27.selftest()
    if a.selftest:
        print("[selftest 통과]")
        return
    RES.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    base, seeds = D27.load_inputs()
    blk = load_block(base)
    cache = build_cache(base, seeds, blk)
    worst = reproduce_gate(cache)
    t = group_tests(cache)
    print("[묶음 검정] 완료", flush=True)
    rk, sd = regime_with_block(cache)
    et, ec = eda_regime(base, a.workers)
    figs = {"group": fig_group(t), "rank": fig_rnn_rank(rk, sd), "eda": fig_eda(et)}
    write_report(t, rk, sd, figs, worst, et, ec, sorted({k[0] for k in base}))
    print(f"[27e 완료] {RES / (STEM + '_report.md')}", flush=True)


if __name__ == "__main__":
    main()
