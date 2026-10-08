# %% [markdown]
# # 27d번: 변동성 국면별 최선·순차 대 병렬 묶음 검정·비정상성 검정(시드 5개, 재학습 없음)
#
# 27번의 저장된 평가 예측(시드 0~4)만 읽는다. 27번 1차 마무리 뒤 Codex 적대적 리뷰(2026-10-08)와 사용자 지적으로
# 빠져 있던 세 가지를 채운다.
#
# 1. **비정상성 검정(20종목)**: 로그가격·로그수익률·예측 구간별 실현변동성(RV, 로그 RV)에 ADF·KPSS, 수익률에
#    ARCH-LM, 학습 대 평가 RV 분포 차이(KS 통계량). 지금까지는 BTC 한 종목(24번)으로만 확인했다.
# 2. **변동성 국면별 최선(시드 5개)**: 국면은 직전 H분 RV의 종목별 학습 구간 5분위(Q1 잔잔 ~ Q5 요동)로 미리
#    나눈다(사후 실현 RV 분할은 예측자의 딜레마로 쓰지 않는다). 26b 3-2절은 시드 0만 썼다. 같은 검정을 시드 평균
#    손실로 다시 하고, 시드마다 최선이 유지되는지 센다. 시드 0으로 26b 3-2절을 그대로 재현하는지 먼저 확인한다.
# 3. **순차 대 병렬 묶음 DM(시드 5개)**: 시점을 차례로 처리하며 상태를 이어받는 재귀형(순차)과 창 전체를
#    한꺼번에 처리하는 형(병렬)의 묶음 평균 손실을 예측 구간 전체와 국면별로 비교한다.
#
# 분류 기준: window slicing(창 잘라 넣기)은 거의 모든 모델의 공통 전처리라 기준이 될 수 없다. 기준은 "창 안의
# 시점을 차례로 처리하며 상태를 이어받는가"다. GARCH+LightGBM은 GARCH 예측을 특징으로 쓰는 트리라 병렬(특징
# 기반)에 두고 혼합형으로 표시한다. S-Mamba는 시간축을 선형 임베딩으로 한꺼번에 처리하고 Mamba를 변수 축에 써서
# 시간 기준으로는 병렬이다.

# %%
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
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
        if (p / "engine").is_dir() and (p / "AGENTS.md").exists():
            return p
    return start


ROOT = _project_root(Path(__file__).resolve())
TAG = "27d_regime_seqpar_20261008"
STEM = "27d_regime_seqpar"
RES = ROOT / "test" / "results" / TAG
IMG = ROOT / "test" / "images" / TAG


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


X = _load("x27_for27d", ROOT / "test" / "models" / "27_model_expansion_test.py")
M = X.M
os.environ.setdefault("RUN26B_SRC", "27")
B26 = _load("b26_for27d", ROOT / "test" / "models" / "26b_robustness_significance_test.py")
R27 = X.RES
R26B = ROOT / "test" / "results" / "26b_robust_signif_27_20261005"

SEEDS = (0, 1, 2, 3, 4)
ALPHA = 0.05
SEQ = ("GARCH-t", "MS-GARCH", "TAR-GARCH", "GRU", "LSTM")
PAR_FEAT = ("LightGBM", "XGBoost", "HistGBM", "GARCH+LightGBM", "Nystroem+Ridge", "KernelRidge-RBF", "SVR-RBF")
PAR_DEEP = ("PatchTST", "iTransformer", "Autoformer", "TimeXer", "TCN", "TimesNet", "ModernTCN", "S-Mamba") + tuple(X.FM_MODELS)
GROUP_OF = {**{m: "순차" for m in SEQ}, **{m: "병렬(특징 기반)" for m in PAR_FEAT}, **{m: "병렬(딥러닝)" for m in PAR_DEEP}}
GROUP_ORDER = ("순차", "병렬(특징 기반)", "병렬(딥러닝)")
# 고정 색 순서(27b 팔레트에서 가져옴). 색은 묶음을 따라가고 순위를 따라가지 않는다.
# 순차 = 파랑, 병렬 = 주황 계열(특징 기반 주황, 딥러닝 갈색)이라 칸 색만 봐도 순차/병렬이 갈린다.
GROUP_COLOR = {"순차": "#2a78d6", "병렬(특징 기반)": "#eb6834", "병렬(딥러닝)": "#8a6b2e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e5e0"
COMPARISONS = (  # (이름, A 구성원, B 구성원). 손실차 = A − B, 음수면 A(순차 쪽)가 낫다
    ("순차 − 병렬 전체", SEQ, PAR_FEAT + PAR_DEEP),
    ("순차 − 병렬(특징 기반)", SEQ, PAR_FEAT),
    ("순차 − 병렬(딥러닝)", SEQ, PAR_DEEP),
    ("순환 신경망 − 병렬(딥러닝)", ("GRU", "LSTM"), PAR_DEEP),
)
SCOPES = ("전체", "Q1", "Q2", "Q3", "Q4", "Q5")
SHORT = {**M.SHORT, "KernelRidge-RBF": "KRR", "SVR-RBF": "SVR"}
_LINES: list[str] = []


def emit(t: str = "") -> None:
    _LINES.append(t)


def sname(m: str) -> str:
    return SHORT.get(m, m)


def fp(p: float) -> str:
    return "해당 없음(표본 부족)" if not np.isfinite(p) else (f"{p:.2g}" if p < 0.001 else f"{p:.3f}")


# %% [markdown]
# ## 입력과 게이트

# %%
def load_inputs() -> tuple[dict, dict]:
    base = M._npz_to_store(R27 / "27_model_expansion_test_predictions.npz")
    miss = [s for s in SEEDS[1:] if not (R27 / f"27_model_expansion_seed{s}_test_predictions.npz").exists()]
    if miss:
        raise RuntimeError(f"[시드 완전성 게이트: 27d] 시드 예측 파일 없음: {miss}")
    seeds = {s: M._npz_to_store(R27 / f"27_model_expansion_seed{s}_test_predictions.npz") for s in SEEDS[1:]}
    M.check_seed_preds(base, {s: {k: S["preds"] for k, S in st.items()} for s, st in seeds.items()},
                       X.STOCHASTIC_ALL, "27d")
    # 결정론 확인: 시드 실행에 함께 돌린 GARCH-t는 시드 0과 같아야 한다
    dmax = max(float(np.max(np.abs(st[k]["preds"]["GARCH-t"] - base[k]["preds"]["GARCH-t"])))
               for st in seeds.values() for k in base if "GARCH-t" in st.get(k, {}).get("preds", {}))
    if dmax > 1e-9:
        raise RuntimeError(f"결정론 모델 GARCH-t의 시드 예측이 시드 0과 다르다(최대차 {dmax:.3g})")
    models = sorted({nm for S in base.values() for nm in S["preds"]} - {"naive"})
    unk = [m for m in models if m not in GROUP_OF]
    if unk or len(PAR_DEEP) != 15 or len(set(GROUP_OF)) != len(SEQ) + len(PAR_FEAT) + len(PAR_DEEP):
        raise RuntimeError(f"분류되지 않았거나 중복된 모델: {unk}")
    print(f"[입력] 칸 {len(base)}개, 모델 {len(models)}종, GARCH-t 시드 최대차 {dmax:.1g}", flush=True)
    return base, seeds


def losses_by_seed(base: dict, seeds: dict, H: int) -> tuple[dict, dict, pd.DataFrame]:
    """반환: (시드 → 모델 → (시각 × 종목) QLIKE, 모델 → 시드 평균 QLIKE, (시각 × 종목) 국면 번호 0~4)."""
    Ls = {s: {} for s in SEEDS}
    Q = {}
    keys = sorted(k for k in base if k[1] == H)
    for key in keys:
        S = base[key]
        tk = key[0]
        T = pd.DatetimeIndex(np.asarray(S["T"]).astype("datetime64[ns]"))
        a = S["act"].astype(float)
        Q[tk] = pd.Series(np.digitize(S["nai"], np.quantile(S["nai_tr"], [.2, .4, .6, .8])), index=T)
        for nm, p0 in S["preds"].items():
            l0 = M.qlike_vec(a, p0.astype(float))
            for s in SEEDS:
                if s == 0 or nm not in X.STOCHASTIC_ALL:
                    l = l0                      # 결정론 모델은 시드와 무관(위에서 GARCH-t로 확인)
                else:
                    l = M.qlike_vec(a, seeds[s][key]["preds"][nm].astype(float))
                Ls[s].setdefault(nm, {})[tk] = pd.Series(l, index=T)
    Ls = {s: {nm: pd.DataFrame(c).sort_index() for nm, c in d.items()} for s, d in Ls.items()}
    Lm = {nm: sum(Ls[s][nm] for s in SEEDS) / len(SEEDS) for nm in Ls[0]}
    return Ls, Lm, pd.DataFrame(Q).sort_index()


def scope_mask(Qf: pd.DataFrame, scope: str) -> pd.DataFrame:
    return Qf.notna() if scope == "전체" else (Qf == int(scope[1]) - 1)


# %% [markdown]
# ## 국면별 최선과 동률(26b 3-2절과 같은 정의)

# %%
def best_and_ties(L: dict, mask: pd.DataFrame) -> pd.DataFrame:
    """국면 최선 대비 열세 검정. 시각마다 그 국면에 든 종목의 손실차 평균 → HAC, 모델 전체에 Holm."""
    mean_l = {m: L[m].where(mask.reindex_like(L[m]).fillna(False)).stack().mean() for m in L}
    best = min((m for m in mean_l if m != "naive"), key=lambda m: mean_l[m])
    rows = []
    for nm in L:
        if nm == best:
            continue
        d = (L[nm] - L[best]).where(mask.reindex_like(L[nm]).fillna(False)).mean(axis=1).dropna().to_numpy()
        mu, se, p = B26.hac_mean_test(d)
        rows.append({"모델": nm, "최선": best, "격차": mu, "표준오차": se, "p": p, "시각수": len(d)})
    t = pd.DataFrame(rows)
    t["p_holm"] = B26.holm(t["p"].fillna(1).to_numpy())
    t["통계적동률"] = t["p_holm"] > ALPHA
    t = pd.concat([t, pd.DataFrame([{"모델": best, "최선": best, "격차": 0.0, "표준오차": 0.0, "p": 1.0,
                                     "시각수": int(t["시각수"].max()), "p_holm": 1.0, "통계적동률": True}])], ignore_index=True)
    return t


def regime_analysis(base: dict, seeds: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, seed_rows, cache = [], [], {}
    ref = pd.read_csv(R26B / "26b_robust_signif_27_regime_significance.csv")
    worst = 0.0
    for H in M.HORIZONS_H:
        Ls, Lm, Qf = losses_by_seed(base, seeds, H)
        cache[H] = (Ls, Lm, Qf)
        for scope in SCOPES:
            mask = scope_mask(Qf, scope)
            nobs = int(mask.sum().sum())
            t = best_and_ties(Lm, mask).assign(H=H, 국면=scope, 관측수=nobs)
            rows.append(t)
            for s in SEEDS:
                ts = best_and_ties(Ls[s], mask)
                seed_rows.append(ts.assign(H=H, 국면=scope, 시드=s)[["H", "국면", "시드", "모델", "최선", "통계적동률"]])
                if s == 0 and scope != "전체":   # 재현 게이트: 시드 0은 26b 3-2절과 같아야 한다
                    r = ref[(ref["H"] == H) & (ref["구간"] == scope)].set_index("모델")
                    g = ts.set_index("모델")
                    if r["최선"].iloc[0] != g["최선"].iloc[0] or set(r.index) != set(g.index) \
                            or not (r["통계적동률"] == g.loc[r.index, "통계적동률"]).all():
                        raise RuntimeError(f"[재현 게이트] 시드 0 {H} {scope}가 26b 3-2절과 다르다")
                    worst = max(worst, float(np.nanmax(np.abs(r["p_holm"] - g.loc[r.index, "p_holm"]))))
        print(f"  [국면] {M.hlabel(H)} 완료", flush=True)
    if worst > 1e-9:
        raise RuntimeError(f"[재현 게이트] 시드 0 p_holm이 26b와 다르다(최대차 {worst:.3g})")
    print(f"[재현 게이트] 시드 0 국면 검정이 26b 3-2절과 일치(p_holm 최대차 {worst:.1g})", flush=True)
    rg = pd.concat(rows, ignore_index=True)
    sd = pd.concat(seed_rows, ignore_index=True)
    rg.to_csv(RES / f"{STEM}_regime_best_ties.csv", index=False)
    sd.to_csv(RES / f"{STEM}_regime_best_ties_by_seed.csv", index=False)
    return rg, sd, cache


# %% [markdown]
# ## 순차 대 병렬 묶음 DM

# %%
def group_frame(L: dict, members: tuple) -> pd.DataFrame:
    have = [m for m in members if m in L]
    return sum(L[m] for m in have) / len(have)


def seqpar_tests(cache: dict) -> pd.DataFrame:
    rows = []
    for name, A, Bm in COMPARISONS:
        for H in M.HORIZONS_H:
            Ls, Lm, Qf = cache[H]
            nA, nB = len([m for m in A if m in Lm]), len([m for m in Bm if m in Lm])
            for scope in SCOPES:
                mask = scope_mask(Qf, scope)
                d = (group_frame(Lm, A) - group_frame(Lm, Bm)).where(mask).mean(axis=1).dropna().to_numpy()
                mu, se, p = B26.hac_mean_test(d)
                r = {"비교": name, "H": H, "국면": scope, "평균차": mu, "표준오차": se, "p": p, "시각수": len(d),
                     "A구성원수": nA, "B구성원수": nB}
                for s in SEEDS:
                    ds = (group_frame(Ls[s], A) - group_frame(Ls[s], Bm)).where(mask).mean(axis=1).dropna().to_numpy()
                    r[f"시드{s}_평균차"], _, r[f"시드{s}_p"] = B26.hac_mean_test(ds)
                rows.append(r)
    t = pd.DataFrame(rows)
    out = []
    for name, g in t.groupby("비교", sort=False):     # Holm은 비교마다 30칸(5구간 × 6범위)에 건다
        g = g.copy()
        g["p_holm"] = B26.holm(g["p"].fillna(1).to_numpy())
        same, opp = np.zeros(len(g), int), np.zeros(len(g), int)
        for s in SEEDS:
            ph = B26.holm(g[f"시드{s}_p"].fillna(1).to_numpy())
            sig = ph < ALPHA
            same += sig & (np.sign(g[f"시드{s}_평균차"].to_numpy()) == np.sign(g["평균차"].to_numpy()))
            opp += sig & (np.sign(g[f"시드{s}_평균차"].to_numpy()) != np.sign(g["평균차"].to_numpy()))
        g["같은방향유의_시드수"], g["반대방향유의_시드수"] = same, opp
        g["판정"] = np.where(g["p_holm"] >= ALPHA, "구분 안 됨", np.where(g["평균차"] < 0, "순차 우세", "병렬 우세"))
        out.append(g)
    t = pd.concat(out, ignore_index=True)
    t.to_csv(RES / f"{STEM}_seqpar_dm.csv", index=False)
    return t


# %% [markdown]
# ## 비정상성 검정(20종목, 종목마다 프로세스 하나)

# %%
NS_SERIES = ["로그가격", "로그수익률"] + [f"RV {M.hlabel(H)}" for H in M.HORIZONS_H] + [f"로그RV {M.hlabel(H)}" for H in M.HORIZONS_H]


def rv_blocks(D: dict, H: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """자정에 맞춘 겹치지 않는 H분 블록의 RV(원 수익률 기준, 0 포함). 결측 봉이 든 블록은 뺀다."""
    g, m = D["grid"], H // 15
    off = int(round((g[0] - g[0].floor("D")) / pd.Timedelta("15min"))) % m
    a0 = (-off) % m
    nb = (len(g) - a0) // m
    lo = a0 + np.arange(nb) * m
    r = np.nan_to_num(D["r"], nan=0.0)
    cs = np.r_[0.0, np.cumsum(r ** 2)]
    cb = np.r_[0.0, np.cumsum(D["bad"].astype(float))]
    rv = np.sqrt(np.maximum(cs[lo + m] - cs[lo], 0.0))
    ok = (cb[lo + m] - cb[lo]) == 0
    return pd.DatetimeIndex(g[lo])[ok], rv[ok]


def _unitroot(x: np.ndarray) -> dict:
    from statsmodels.tsa.stattools import adfuller, kpss
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    lag = int(12 * (n / 100) ** 0.25)          # Schwert 규칙의 고정 시차(자동 선택은 10만 행에서 비정상 종료 이력)
    adf_p = float(adfuller(x, maxlag=lag, regression="c", autolag=None)[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = kpss(x, regression="c", nlags="auto")
    kpss_p = float(kp[1])
    if adf_p < ALPHA and kpss_p > ALPHA:
        v = "정상"
    elif adf_p >= ALPHA and kpss_p <= ALPHA:
        v = "비정상(단위근)"
    elif adf_p < ALPHA and kpss_p <= ALPHA:
        v = "엇갈림(장기기억·구조 변화)"
    else:
        v = "판정 불가"
    return {"n": n, "ADF_p": adf_p, "KPSS_p": kpss_p, "판정": v}


def ns_job(tk: str) -> list[dict]:
    from scipy.stats import ks_2samp
    from statsmodels.stats.diagnostic import het_arch
    D = M.build_data(tk)
    r = D["r"]
    ok = np.isfinite(r)
    rows = [{"종목": tk, "계열": "로그가격", **_unitroot(np.cumsum(np.where(ok, r, 0.0)))}]
    u = _unitroot(r[ok])
    lm = het_arch(r[ok], nlags=16)
    rows.append({"종목": tk, "계열": "로그수익률", **u, "ARCH_LM_p": float(lm[1])})
    for H in M.HORIZONS_H:
        T, rv = rv_blocks(D, H)
        rows.append({"종목": tk, "계열": f"RV {M.hlabel(H)}", **_unitroot(rv)})
        pos = rv > 0
        rows.append({"종목": tk, "계열": f"로그RV {M.hlabel(H)}", **_unitroot(np.log(rv[pos]))})
        tr, te = np.asarray(T < M.SPLIT), np.asarray(T >= M.SPLIT)
        ks = ks_2samp(rv[tr], rv[te])
        rows[-1].update({"KS_D": float(ks.statistic), "정지비율_학습": float(np.mean(rv[tr] == 0)),
                         "정지비율_평가": float(np.mean(rv[te] == 0)),
                         "로그RV평균_학습": float(np.log(rv[tr & pos]).mean()), "로그RV평균_평가": float(np.log(rv[te & pos]).mean())})
    return rows


def nonstationarity(tickers: list[str], workers: int) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for rs in ex.map(ns_job, tickers) for r in rs]
    ns = pd.DataFrame(rows)
    ns.to_csv(RES / f"{STEM}_nonstationarity.csv", index=False)
    return ns


def seed_keep(sd: pd.DataFrame, H: int, scope: str, best: str) -> tuple[int, int]:
    """(같은 모델이 최선인 시드 수, 같은 처리 방식 묶음의 모델이 최선인 시드 수)."""
    b = sd[(sd["H"] == H) & (sd["국면"] == scope) & (sd["모델"] == sd["최선"])]["최선"]
    return int((b == best).sum()), int(b.map(GROUP_OF).eq(GROUP_OF[best]).sum())


# %% [markdown]
# ## 그림

# %%
def fig_regime_map(rg: pd.DataFrame, sd: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    Hs = list(M.HORIZONS_H)
    for i, scope in enumerate(SCOPES):
        for j, H in enumerate(Hs):
            g = rg[(rg["H"] == H) & (rg["국면"] == scope)]
            best = g["최선"].iloc[0]
            col = GROUP_COLOR[GROUP_OF[best]]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=col, alpha=0.22 if scope != "전체" else 0.32,
                                       edgecolor="white", lw=3))
            k, kg = seed_keep(sd, H, scope, best)
            tie = set(g.loc[g["통계적동률"], "모델"])
            n_new = len(tie & set(PAR_DEEP))
            n_all = len(set(g["모델"]) & set(PAR_DEEP))
            seq_in = len(tie & set(SEQ))
            ax.text(j + 0.5, i + 0.36, sname(best) + (" *" if best == "GARCH+LightGBM" else ""), ha="center",
                    va="center", fontsize=12.5, fontweight="bold", color=INK)
            ax.text(j + 0.5, i + 0.62, f"시드 최선: 모델 {k}/5 · 방식 {kg}/5", ha="center", va="center", fontsize=9.5, color=MUTED)
            ax.text(j + 0.5, i + 0.83, f"동률: 순차 {seq_in}/5 · 병렬 딥러닝 {n_new}/{n_all}", ha="center",
                    va="center", fontsize=9, color=MUTED)
    ax.set_xlim(0, len(Hs))
    ax.set_ylim(len(SCOPES), 0)
    ax.set_xticks(np.arange(len(Hs)) + 0.5, [M.hlabel(H) for H in Hs], fontsize=12)
    ax.set_yticks(np.arange(len(SCOPES)) + 0.5,
                  ["전체 기간", "Q1 (가장 잔잔)", "Q2", "Q3", "Q4", "Q5 (가장 요동)"], fontsize=12)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(1, color=INK, lw=1.2)
    ax.set_xlabel("예측 구간", fontsize=12)
    ax.set_ylabel("직전 변동성 국면(종목별 학습 구간 5분위)", fontsize=12)
    ax.set_title("예측 구간 × 변동성 국면별 최선 모델(시드 5개 평균 QLIKE)\n칸 색 = 최선 모델의 처리 방식 · 시드 최선 = 시드별로 다시 골랐을 때 같은 모델 / 같은 처리 방식이 최선인 횟수", fontsize=13,
                 loc="left", color=INK)
    ax.legend(handles=[Patch(facecolor=GROUP_COLOR[g_], alpha=0.5, label=lb) for g_, lb in
                       zip(GROUP_ORDER, ["순차(재귀형: GARCH·GRU·LSTM)", "병렬(특징 기반: 트리·커널, * GARCH 특징을 쓰는 혼합형)",
                                         "병렬(딥러닝: 어텐션·합성곱·파운데이션)"])],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1, frameon=False, fontsize=10.5)
    fig.tight_layout()
    p = IMG / f"{STEM}_fig1_regime_best_map.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_seqpar(t: pd.DataFrame) -> Path:
    cat_col = {"순차 우세": GROUP_COLOR["순차"], "병렬 우세": "#eb6834", "구분 안 됨": GRID}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    Hs = list(M.HORIZONS_H)
    for ax, (name, _, _) in zip(axes.ravel(), COMPARISONS):
        g = t[t["비교"] == name]
        for i, scope in enumerate(SCOPES):
            for j, H in enumerate(Hs):
                x = g[(g["H"] == H) & (g["국면"] == scope)].iloc[0]
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=cat_col[x["판정"]],
                                           alpha=0.75 if x["판정"] != "구분 안 됨" else 1.0, edgecolor="white", lw=2))
                tc = "white" if x["판정"] != "구분 안 됨" else INK
                ax.text(j + 0.5, i + 0.42, f"{x['평균차']:+.3f}", ha="center", va="center", fontsize=10.5, color=tc,
                        fontweight="bold")
                ax.text(j + 0.5, i + 0.72, f"시드 {int(x['같은방향유의_시드수'])}/5", ha="center", va="center", fontsize=8.5,
                        color=tc)
        ax.set_xlim(0, len(Hs))
        ax.set_ylim(len(SCOPES), 0)
        ax.set_xticks(np.arange(len(Hs)) + 0.5, [M.hlabel(H) for H in Hs], fontsize=10.5)
        ax.set_yticks(np.arange(len(SCOPES)) + 0.5, ["전체 기간", "Q1 잔잔", "Q2", "Q3", "Q4", "Q5 요동"], fontsize=10.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(1, color=INK, lw=1.0)
        ax.set_title(f"{name}  (손실차, 음수 = 순차 쪽이 낫다)", fontsize=12, loc="left", color=INK)
    fig.legend(handles=[Patch(facecolor=cat_col[k], label=k + (" (Holm p<0.05)" if k != "구분 안 됨" else " (Holm p≥0.05)"))
                        for k in cat_col], loc="lower center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("순차(재귀형) 대 병렬 묶음 평균 손실 DM 검정(시드 5개 평균, 예측 구간 × 변동성 국면)\n"
                 "칸 아래 숫자 = 시드별로 다시 검정해 같은 방향으로 유의했던 시드 수", fontsize=13.5, x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    p = IMG / f"{STEM}_fig2_seqpar_dm.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_nonstationarity(ns: pd.DataFrame) -> Path:
    cats = ["정상", "엇갈림(장기기억·구조 변화)", "비정상(단위근)", "판정 불가"]
    col = {"정상": "#1baf7a", "엇갈림(장기기억·구조 변화)": "#eda100", "비정상(단위근)": "#d0342c", "판정 불가": GRID}
    tks = sorted(ns["종목"].unique(), key=lambda s: s.replace("KRW-", ""))
    fig, ax = plt.subplots(figsize=(14, 9.5))
    for i, tk in enumerate(tks):
        for j, se in enumerate(NS_SERIES):
            v = ns[(ns["종목"] == tk) & (ns["계열"] == se)]["판정"].iloc[0]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=col[v], edgecolor="white", lw=1.5))
    ax.set_xlim(0, len(NS_SERIES))
    ax.set_ylim(len(tks), 0)
    ax.set_xticks(np.arange(len(NS_SERIES)) + 0.5, NS_SERIES, rotation=40, ha="right", fontsize=10.5)
    ax.set_yticks(np.arange(len(tks)) + 0.5, [t_.replace("KRW-", "") for t_ in tks], fontsize=10.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for x in (1, 2, 7):
        ax.axvline(x, color=INK, lw=1.2)
    ax.set_title("20종목 단위근·정상성 검정(ADF·KPSS, 유의수준 0.05, 전체 기간 2023-10~2026-10)", fontsize=13.5,
                 loc="left", color=INK)
    ax.legend(handles=[Patch(facecolor=col[c], label=c) for c in cats], loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=4, frameon=False, fontsize=11)
    fig.tight_layout()
    p = IMG / f"{STEM}_fig3_nonstationarity.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_shift(ns: pd.DataFrame) -> Path:
    g = ns[ns["계열"].str.startswith("로그RV")].copy()
    tks = sorted(g["종목"].unique(), key=lambda s: s.replace("KRW-", ""))
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5))
    pv = g.pivot(index="종목", columns="계열", values="KS_D").loc[tks, [f"로그RV {M.hlabel(H)}" for H in M.HORIZONS_H]]
    im = axes[0].imshow(pv.to_numpy(), cmap="Blues", vmin=0, vmax=max(0.5, float(pv.max().max())), aspect="auto")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.iat[i, j]
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v > 0.3 else INK)
    axes[0].set_xticks(range(pv.shape[1]), [M.hlabel(H) for H in M.HORIZONS_H])
    axes[0].set_yticks(range(len(tks)), [t_.replace("KRW-", "") for t_ in tks])
    axes[0].set_title("학습 대 평가 RV 분포 차이(KS 통계량 D)\n0 = 같은 분포, 클수록 분포가 바뀜", fontsize=12, loc="left")
    fig.colorbar(im, ax=axes[0], fraction=0.04, label="KS D")
    d15 = g[g["계열"] == f"로그RV {M.hlabel(15)}"].set_index("종목").loc[tks]
    y = np.arange(len(tks))
    axes[1].hlines(y, d15["정지비율_학습"] * 100, d15["정지비율_평가"] * 100, color=GRID, lw=3)
    axes[1].scatter(d15["정지비율_학습"] * 100, y, color="#2a78d6", s=55, label="학습 구간", zorder=3)
    axes[1].scatter(d15["정지비율_평가"] * 100, y, color="#eb6834", s=55, label="평가 구간", zorder=3)
    axes[1].set_yticks(y, [t_.replace("KRW-", "") for t_ in tks])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("15분 블록 중 가격 정지(RV=0) 비율(%)")
    axes[1].set_title("15분 가격 정지 비율: 학습 → 평가", fontsize=12, loc="left")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(axis="x", color=GRID)
    for s in ("top", "right"):
        axes[1].spines[s].set_visible(False)
    fig.tight_layout()
    p = IMG / f"{STEM}_fig4_distribution_shift.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


# %% [markdown]
# ## 보고서

# %%
def write_report(ns: pd.DataFrame, rg: pd.DataFrame, sd: pd.DataFrame, t: pd.DataFrame, figs: dict) -> None:
    rel = lambda p: os.path.relpath(p, RES)
    emit("# 27d번: 변동성 국면별 최선·순차 대 병렬 묶음 검정·비정상성 검정(시드 5개)")
    emit()
    emit("27번의 저장된 평가 예측(시드 0~4)만 읽고 재학습하지 않았다. 27번 1차 마무리 원고에서 빠져 있던 변동성 국면 결과와 "
         "비정상성 근거를 채우고, 26b 3-2절(시드 0만 사용)의 국면 검정을 시드 5개로 다시 했다.")
    emit()
    emit("## 0. 설계와 읽는 법")
    emit()
    emit("- **변동성 국면**: 직전 H분 RV를 그 종목의 학습 구간 5분위로 나눈다(Q1 가장 잔잔 ~ Q5 가장 요동). 예측 시점에 이미 아는 "
         "값으로 나누므로 실전에서도 쓸 수 있다. 24·25번처럼 실제(사후) RV로 나누면 낮게 예측하는 모델이 저변동 구간에서 이기는 "
         "착시(예측자의 딜레마, Lerch 외 2017)가 생겨 쓰지 않는다.")
    emit("- **처리 방식 분류**: window slicing은 거의 모든 모델의 공통 전처리라 기준이 아니다. 기준은 창 안의 시점을 차례로 처리하며 "
         "상태를 이어받는가다.")
    emit(f"  - 순차(재귀형) {len(SEQ)}종: {', '.join(SEQ)}")
    emit(f"  - 병렬(특징 기반) {len(PAR_FEAT)}종: {', '.join(PAR_FEAT)}. GARCH+LightGBM은 GARCH 예측을 특징으로 쓰는 혼합형이다.")
    emit(f"  - 병렬(딥러닝) {len(PAR_DEEP)}종: {', '.join(PAR_DEEP)}. S-Mamba는 시간축을 한꺼번에 임베딩하고 Mamba를 변수 축에 쓴다.")
    emit("- **손실**: QLIKE. 무작위 모델은 시드마다 그 시드의 예측으로 손실을 구해 평균했다(손실의 평균이지 예측의 평균이 아니다). "
         "결정론 모델은 시드와 무관하다(GARCH-t 시드 예측이 시드 0과 같음을 확인).")
    emit("- **국면 최선 검정**: 비교 집단은 국면 최선 모델 대 나머지 각 모델이다. H0: 두 모델의 기대 QLIKE가 같다. H1: 같지 않다(양측). "
         "시각마다 그 국면에 든 종목의 손실차를 평균한 시계열에 DM 검정(Newey-West HAC)을 하고, 모델 전체에 Holm 보정을 건다. "
         "H0를 기각하지 못하면 통계적 동률(최선과 구분되지 않음), 기각하면 열세다.")
    emit("- **순차 대 병렬 묶음 검정**: 비교 집단은 묶음 구성원 손실의 단순 평균끼리다(묶음 안에서 잘한 모델만 고르는 선택 편향을 피한다). "
         "H0: 두 묶음의 기대 QLIKE가 같다. H1: 같지 않다(양측). 비교마다 30칸(예측 구간 5 × 범위 6)에 Holm 보정을 건다. "
         "시드마다 같은 검정을 다시 해 같은 방향으로 유의했던 시드 수를 함께 적는다.")
    emit("- **게이트**: 시드 1~4 예측 파일과 무작위 모델 17종의 예측이 모두 있어야 하고, 시드 0으로 돌린 국면 검정이 26b 3-2절을 "
         "그대로 재현해야(최선·동률 판정 일치, p 일치) 이후 결과를 만든다. 이번 실행은 두 게이트를 모두 통과했다.")
    emit()

    # 1. 비정상성
    emit("## 1. 데이터는 비정상인가(20종목)")
    emit()
    emit(f"![비정상성 검정]({rel(figs['ns'])})")
    emit()
    emit("**읽는 법**: 행은 종목, 열은 시계열이다. ADF의 H0는 \"단위근이 있다(비정상)\", KPSS의 H0는 \"정상이다\"로 귀무가설이 서로 반대라 "
         "함께 본다. ADF 기각·KPSS 기각 못함 = 정상(초록), ADF 기각 못함·KPSS 기각 = 비정상(빨강), 둘 다 기각 = 엇갈림(노랑, 장기기억이나 "
         "수준 이동이 있을 때 흔하다), 둘 다 기각 못함 = 판정 불가(회색). RV는 가격 정지(0)를 포함한 값이고, 로그 RV는 0을 뺀 값이다.")
    emit()
    vc = ns.pivot_table(index="계열", columns="판정", values="종목", aggfunc="count").reindex(NS_SERIES).fillna(0).astype(int)
    cols = [c for c in ["정상", "엇갈림(장기기억·구조 변화)", "비정상(단위근)", "판정 불가"]]
    for c in cols:
        if c not in vc.columns:
            vc[c] = 0
    emit("| 시계열 | " + " | ".join(cols) + " | ADF p 중앙값 | KPSS p 중앙값 |")
    emit("| :--- | " + " | ".join(["---:"] * (len(cols) + 2)) + " |")
    for se in NS_SERIES:
        g = ns[ns["계열"] == se]
        emit(f"| {se} | " + " | ".join(str(int(vc.loc[se, c])) for c in cols)
             + f" | {g['ADF_p'].median():.3f} | {g['KPSS_p'].median():.3f} |")
    emit()
    arch = ns[ns["계열"] == "로그수익률"]
    lp = ns[ns["계열"] == "로그가격"]
    emit(f"- **로그가격**: 20종목 중 {int((lp['판정'] == '비정상(단위근)').sum())}종목이 비정상(단위근)이다"
         f"(판정 분포: {', '.join(f'{k} {v}' for k, v in lp['판정'].value_counts().items())}). 가격 수준을 그대로 예측 대상으로 쓰면 안 되는 이유다.")
    emit(f"- **로그수익률**: 평균 수준은 {int((arch['판정'] == '정상').sum())}종목이 정상이다. 그러나 ARCH-LM(H0: 조건부 이분산이 없다) p가 "
         f"0.05 미만인 종목이 {int((arch['ARCH_LM_p'] < ALPHA).sum())}/20개(최댓값 {arch['ARCH_LM_p'].max():.2g})다. 곧 수익률의 분산이 시간에 따라 "
         "변하고 변동성 군집이 있다. 이 회차의 예측 대상(변동성)이 바로 이 시간에 따라 변하는 분산이다.")
    for kind in ("RV", "로그RV"):
        parts = []
        for H in M.HORIZONS_H:
            g = ns[ns["계열"] == f"{kind} {M.hlabel(H)}"]["판정"].value_counts()
            parts.append(f"{M.hlabel(H)} " + "·".join(f"{k.split('(')[0]} {v}" for k, v in g.items()))
        emit(f"- **{kind}**: " + "; ".join(parts) + ".")
    mix = {se: int((ns[ns["계열"] == se]["판정"] == "엇갈림(장기기억·구조 변화)").sum()) for se in NS_SERIES}
    stat = {se: int((ns[ns["계열"] == se]["판정"] == "정상").sum()) for se in NS_SERIES}
    odd = lp[lp["판정"] != "비정상(단위근)"]
    emit(f"- **해석**: 가격은 비정상, 수익률은 평균만 정상이고 분산은 시간에 따라 변한다. 예측 대상인 변동성(로그 RV)은 단위근 비정상으로 판정된 "
         f"종목이 하나도 없고, 엇갈림이 15분 {mix['로그RV 15분']}/20 → 12시간 {mix['로그RV 12시간']}/20종목이다. 엇갈림은 ADF가 단위근을 "
         "기각(평균으로 돌아오는 힘이 있음)하면서 KPSS도 정상성을 기각(고정된 평균·분산으로 설명되지 않음)한 경우로, 변동성의 장기기억(매우 느린 "
         "평균 회귀)이나 수준 이동이 있을 때 나타난다(Andersen 외 2003, Corsi 2009의 HAR 모형이 다루는 성질). 곧 이 데이터의 변동성은 \"차분하면 "
         "끝나는 비정상\"이 아니라 \"평균은 있지만 국면이 오래 지속되고 바뀌는 비정상\"이다. 예측 구간이 길어질수록 정상 판정이 늘어(로그 RV "
         f"15분 {stat['로그RV 15분']}종목 → 12시간 {stat['로그RV 12시간']}종목) 집계가 길수록 잡음이 평균되어 안정된다."
         + (f" 로그가격에서 {', '.join(t_.replace('KRW-', '') for t_ in odd['종목'])}는 엇갈림으로 나왔는데, 3년 창 안에서 가격이 오른 뒤 되돌아와 "
            "ADF가 기각한 것으로 KPSS는 여전히 비정상을 가리킨다." if len(odd) else ""))
    emit()
    emit(f"![분포 이동]({rel(figs['shift'])})")
    emit()
    emit("**읽는 법**: 왼쪽은 종목 × 예측 구간마다 학습 구간과 평가 구간의 로그 RV 분포가 얼마나 다른지(KS 통계량 D, 0~1)다. 시계열은 자기상관이 "
         "있어 KS 검정의 p값(독립 가정)은 과소평가되므로 p 대신 D를 효과 크기로 읽는다. 오른쪽은 15분 블록의 가격 정지 비율이 학습에서 평가로 "
         "어떻게 바뀌었는지다.")
    emit()
    s15 = ns[ns["계열"] == f"로그RV {M.hlabel(15)}"]
    emit("| 예측 구간 | KS D 중앙값 | KS D 최댓값(종목) | 로그 RV 평균 변화 중앙값(평가 − 학습) |")
    emit("| :--- | ---: | :--- | ---: |")
    for H in M.HORIZONS_H:
        g = ns[ns["계열"] == f"로그RV {M.hlabel(H)}"]
        w = g.loc[g["KS_D"].idxmax()]
        emit(f"| {M.hlabel(H)} | {g['KS_D'].median():.3f} | {w['KS_D']:.3f}({w['종목'].replace('KRW-', '')}) | "
             f"{(g['로그RV평균_평가'] - g['로그RV평균_학습']).median():+.3f} |")
    emit()
    up = int((s15["정지비율_평가"] > s15["정지비율_학습"]).sum())
    emit(f"- **해석**: 15분 가격 정지 비율은 {up}/20종목에서 평가 구간이 더 높다(중앙값 학습 {s15['정지비율_학습'].median():.1%} → "
         f"평가 {s15['정지비율_평가'].median():.1%}). 학습 구간에서 배운 분포가 평가 구간에 그대로 이어지지 않는다(분포 이동). "
         "비정상 데이터에서는 이 이동을 따라가는 능력이 예측 성능을 가른다.")
    emit()

    # 2. 국면별 최선
    emit("## 2. 변동성 국면별 최선 모델(시드 5개)")
    emit()
    emit(f"![국면별 최선 지도]({rel(figs['map'])})")
    emit()
    emit("**읽는 법**: 칸마다 시드 5개 평균 QLIKE가 가장 작은 모델(굵은 글씨)이고, 칸 색은 그 모델의 처리 방식이다. \"시드별 최선: 같은 모델 k/5 · 같은 처리 방식 m/5\"는 시드마다 "
         "따로 골랐을 때 같은 모델이 최선이었던 횟수와 같은 처리 방식 묶음의 모델이 최선이었던 횟수, \"동률\"은 최선과 통계적으로 구분되지 않은(Holm p ≥ 0.05) 모델 수를 묶음별로 센 것이다. "
         "맨 윗줄(전체 기간)은 국면을 나누지 않은 결과다. 이 줄은 (시각 × 종목) 손실 전체의 평균으로 최선을 고르므로, 종목별 평균을 "
         "다시 평균한 27번 보고서의 순위와 상장폐지 종목(평가 시각 수가 적음) 때문에 드물게 다를 수 있다.")
    emit()
    emit("| 예측 구간 | 국면 | 관측 수(시각×종목) | 최선(처리 방식) | 시드별 최선: 같은 모델 · 같은 처리 방식 | 통계적 동률 | 동률 중 순차 | 동률 중 병렬(딥러닝) |")
    emit("| :--- | :--- | ---: | :--- | ---: | :--- | ---: | ---: |")
    for H in M.HORIZONS_H:
        for scope in SCOPES:
            g = rg[(rg["H"] == H) & (rg["국면"] == scope)]
            b = g["최선"].iloc[0]
            k, kg = seed_keep(sd, H, scope, b)
            tie = g[g["통계적동률"] & (g["모델"] != b)].sort_values("격차")["모델"].tolist()
            tset = set(tie) | {b}
            n_dp = len(set(g["모델"]) & set(PAR_DEEP))
            emit(f"| {M.hlabel(H)} | {scope} | {int(g['관측수'].iloc[0]):,} | {b}({GROUP_OF[b]}) | {k}/5 · {kg}/5 | "
                 f"{', '.join(sname(x) for x in tie) or '없음(최선만 남음)'} | {len(tset & set(SEQ))}/{len(SEQ)} | {len(tset & set(PAR_DEEP))}/{n_dp} |")
    emit()
    emit("**해석(국면마다 H0: 최선과 기대 QLIKE가 같다, Holm 보정)**:")
    emit()
    for scope in SCOPES:
        parts = []
        for H in M.HORIZONS_H:
            g = rg[(rg["H"] == H) & (rg["국면"] == scope)]
            b = g["최선"].iloc[0]
            tset = set(g.loc[g["통계적동률"], "모델"])
            parts.append(f"{M.hlabel(H)} {b}({GROUP_OF[b]}, 병렬 딥러닝 동률 {len(tset & set(PAR_DEEP))}개)")
        emit(f"- **{scope if scope != '전체' else '전체 기간'}**: " + "; ".join(parts) + ".")
    q5 = rg[(rg["국면"] == "Q5")]
    q5_dp = sum(len(set(g.loc[g["통계적동률"], "모델"]) & set(PAR_DEEP)) for _, g in q5.groupby("H"))
    q5_seq = [M.hlabel(H) for H, g in q5.groupby("H") if GROUP_OF[g["최선"].iloc[0]] == "순차"]
    q5_names = [f"{M.hlabel(H)} {', '.join(sorted(set(g.loc[g['통계적동률'], '모델']) & set(PAR_DEEP)))}" for H, g in q5.groupby("H")
                if set(g.loc[g["통계적동률"], "모델"]) & set(PAR_DEEP)]
    emit(f"- **요동 국면(Q5) 요약**: 다섯 예측 구간을 합쳐 최선과 동률인 병렬 딥러닝 모델은 {q5_dp}개"
         + (f"({'; '.join(q5_names)})" if q5_names else "") + "이고, 나머지 병렬 딥러닝 모델은 모든 예측 구간에서 H0가 기각돼 최선보다 유의하게 "
         f"나빴다. 최선이 순차(재귀형)인 예측 구간은 {', '.join(q5_seq) or '해당 없음(없음)'}이다.")
    emit()
    # 시드 0 대 시드 5개 비교
    ref = pd.read_csv(R26B / "26b_robust_signif_27_regime_significance.csv")
    ch = []
    for H in M.HORIZONS_H:
        for scope in SCOPES[1:]:
            b5 = rg[(rg["H"] == H) & (rg["국면"] == scope)]["최선"].iloc[0]
            b0 = ref[(ref["H"] == H) & (ref["구간"] == scope)]["최선"].iloc[0]
            if b5 != b0:
                ch.append(f"{M.hlabel(H)} {scope}: {b0} → {b5}")
    emit(f"**시드 0(26b 3-2절) 대비 바뀐 최선**: {'; '.join(ch) if ch else '해당 없음(25칸 모두 같은 모델이 최선)'}.")
    emit()

    # 3. 순차 대 병렬
    emit("## 3. 순차(재귀형) 대 병렬 묶음 검정(시드 5개)")
    emit()
    emit(f"![순차 대 병렬]({rel(figs['seqpar'])})")
    emit()
    emit("**읽는 법**: 네 비교마다 행은 범위(전체 기간, 국면 Q1~Q5), 열은 예측 구간이다. 숫자는 묶음 평균 QLIKE 차(A − B)로 음수면 순차 쪽이 낫다. "
         "파랑 = 순차가 유의하게 낫다, 주황 = 병렬이 유의하게 낫다, 회색 = 구분되지 않는다(Holm p ≥ 0.05). \"시드 k/5\"는 시드마다 다시 검정해 "
         "같은 방향으로 유의했던 시드 수다.")
    emit()
    for name, A, Bm in COMPARISONS:
        g = t[t["비교"] == name]
        emit(f"### {name}")
        emit()
        emit(f"비교 집단: A = {', '.join(A)} / B = {len(Bm)}종({'TTM은 4·12시간 평가 제외라 그 구간은 14종' if 'TTM' in Bm else '해당 없음(TTM 미포함)'}). "
             "H0: 두 묶음 평균의 기대 QLIKE가 같다. H1: 같지 않다.")
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
        by = "; ".join(f"{sc if sc != '전체' else '전체 기간'} 순차 {int(((g['국면'] == sc) & (g['판정'] == '순차 우세')).sum())}·병렬 "
                       f"{int(((g['국면'] == sc) & (g['판정'] == '병렬 우세')).sum())}" for sc in SCOPES)
        emit(f"**해석**: 30칸 중 순차 우세 {int(cnt.get('순차 우세', 0))}칸, 병렬 우세 {int(cnt.get('병렬 우세', 0))}칸, 구분 안 됨 "
             f"{int(cnt.get('구분 안 됨', 0))}칸이다. 순차 우세 칸은 H0를 기각하고 순차 묶음의 기대 손실이 작다는 뜻이며, 구분 안 됨은 차이의 증거가 "
             f"부족하다는 뜻이다(같다는 증명은 아니다). 범위별 유의 칸 수(예측 구간 5개 중): {by}.")
        emit()
    emit("**공통 주의**: 순차(GRU·LSTM)와 병렬 딥러닝은 학습 절차(학습률 탐색, 검증 구간, 재적합, 손실 함수)를 맞추지 않았다(27c 참고). "
         "따라서 이 차이를 \"처리 구조만의 효과\"로 단정하지 않고, \"이 학습 절차의 순차 묶음이 낫다/못하다\"로 읽는다. 묶음 평균은 약한 구성원의 "
         "영향을 받으므로 대표 모델끼리의 비교(위 2절의 최선·동률)와 함께 본다.")
    emit()
    emit("## 4. 한계")
    emit()
    emit("1. 국면별 관측 수는 전체의 약 1/5이라 4·12시간 국면 검정은 표본이 작다(2절 표의 관측 수).")
    emit("2. 순차 대 병렬은 학습 절차를 통제하지 않았다(27c와 같은 한계).")
    emit("3. 비정상성 검정은 전체 기간 하나로 했다. 구조 변화 시점 검정은 하지 않았다.")
    emit("4. KS 통계량의 p값은 자기상관 때문에 쓰지 않았고, D를 효과 크기로만 읽었다.")
    (RES / f"{STEM}_report.md").write_text("\n".join(_LINES), encoding="utf-8")


# %%
def selftest() -> None:
    """분류·마스크 synthetic 검사."""
    assert set(SEQ).isdisjoint(PAR_FEAT) and set(SEQ).isdisjoint(PAR_DEEP) and set(PAR_FEAT).isdisjoint(PAR_DEEP)
    Qf = pd.DataFrame({"a": [0, 4, np.nan], "b": [4, 4, 0]})
    assert scope_mask(Qf, "Q5").to_numpy().tolist() == [[False, True], [True, True], [False, False]]
    assert scope_mask(Qf, "전체").to_numpy().sum() == 5
    L = {"x": pd.DataFrame({"a": [1.0, 2.0]}), "y": pd.DataFrame({"a": [3.0, 4.0]})}
    assert np.allclose(group_frame(L, ("x", "y", "없는모델")).to_numpy().ravel(), [2.0, 3.0])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    selftest()
    M.selftest_seed_gate()
    if a.selftest:
        print("[selftest 통과]")
        return
    RES.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    base, seeds = load_inputs()
    tickers = sorted({k[0] for k in base})
    ns = nonstationarity(tickers, a.workers)
    print("[비정상성] 완료", flush=True)
    rg, sd, cache = regime_analysis(base, seeds)
    t = seqpar_tests(cache)
    print("[순차 대 병렬] 완료", flush=True)
    figs = {"ns": fig_nonstationarity(ns), "shift": fig_shift(ns), "map": fig_regime_map(rg, sd), "seqpar": fig_seqpar(t)}
    write_report(ns, rg, sd, t, figs)
    print(f"[27d 완료] {RES / (STEM + '_report.md')}", flush=True)


if __name__ == "__main__":
    main()
