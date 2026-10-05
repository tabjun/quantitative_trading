# %% [markdown]
# # 26b번: 26번 결과의 견고성(시드)·구분 불가 진단·유의성 검정
#
# 26번(시간 기준 재평가)의 예측값을 재적합 없이 읽어 세 가지를 확인한다. 결과는 성능 수치만이 아니라
# "그 결과가 옳은가, 왜 그렇게 나왔는가"를 판정하는 근거로 쓴다.
#
# 1. **시드 견고성(전 모델)**: 무작위성이 있는 모델(트리 4종·Nystroem·GRU·LSTM)은 시드 0~4로 다시 학습한
#    결과의 흔들림을 잰다. 무작위성이 없는 모델은 시드를 바꿔도 예측이 같아야 하므로, 시드 실행에 함께 돌린
#    GARCH-t의 예측이 시드 0과 같은지 확인한다(선형 3종은 결정에 따라 비교에서 제외). MS-GARCH·TAR-GARCH·
#    KernelRidge·SVR은 난수를 쓰지 않는 결정적 알고리즘이라 시드 실행에서 뺐다(코드상 무작위 상태 없음).
# 2. **4시간·12시간 구분 불가 진단**: 모델끼리 차이가 안 나는 이유가 적합 실패인지(조기종료 반복 수, 에폭,
#    보정계수), 데이터 특성인지(평가 표본 수로 본 표준오차, 모델 예측끼리의 유사도, 예측 가능성 상한)를 가른다.
# 3. **유의성 검정**: 26번은 전 종목이 같은 정시 시각을 공유하므로, 시각마다 종목 평균 손실차를 만들면 종목 간
#    상관이 그 시계열에 그대로 담긴다. 이 시계열에 DM 검정(Newey-West HAC)을 하고 다중비교를 Holm으로 보정하며,
#    종목 평균 손실로 MCS를 돌린다. 사전 구간별로는 구간 최선 모델 대비 열세가 유의한지를 같은 방식으로 본다.

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
import numpy as np
import pandas as pd
from scipy import stats as st

matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "engine").is_dir() and (p / "AGENTS.md").exists():
            return p
    return start


ROOT = _project_root(Path(__file__).resolve())
for _p in (ROOT, ROOT / "test" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TAG = "26b_robust_signif_20261005"
STEM = "26b_robust_signif"
IMG = ROOT / "test" / "images" / TAG
RES = ROOT / "test" / "results" / TAG
SRC = ROOT / "test" / "results" / "26_timebased_reeval_20261005"
SRC_STEM = "26_timebased_reeval"


def _load_m26():
    spec = importlib.util.spec_from_file_location("m26_for26b", ROOT / "test" / "models" / "26_timebased_reeval_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M26 = _load_m26()
SEEDS = (0, 1, 2, 3, 4)
STOCHASTIC = ("LightGBM", "XGBoost", "HistGBM", "GARCH+LightGBM", "Nystroem+Ridge", "GRU", "LSTM")
DET_CHECK = ("GARCH-t",)
DET_NOT_RERUN = ("MS-GARCH", "TAR-GARCH", "KernelRidge-RBF", "SVR-RBF")
ALPHA = 0.05
TIE = M26.TIE
_LINES: list[str] = []


def emit(t: str = "") -> None:
    print(t, flush=True)
    _LINES.append(t)


def hac_mean_test(d: np.ndarray) -> tuple[float, float, float]:
    """평균 0 귀무가설의 t 검정(Newey-West HAC, Bartlett 가중). 반환: (평균, 표준오차, 양측 p)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return np.nan, np.nan, np.nan
    mu = d.mean()
    e = d - mu
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(max(v, 1e-30) / n)
    t = mu / se
    return float(mu), float(se), float(2 * st.t.sf(abs(t), df=n - 1))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


# %% [markdown]
# ## 입력

# %%
def load_main() -> tuple[pd.DataFrame, dict]:
    rd = pd.read_csv(SRC / f"{SRC_STEM}_model_comparison.csv")
    z = np.load(SRC / f"{SRC_STEM}_test_predictions.npz")
    store: dict = {}
    names = {"실제": "act", "naive_입력": "nai", "시각": "T", "학습naive": "nai_tr"}
    for k in z.files:
        tk, H, nm = k.split("|")
        S = store.setdefault((tk, int(H)), {"preds": {}})
        if nm in names:
            S[names[nm]] = z[k]
        else:
            S["preds"][nm] = z[k]
    return rd, store


def load_seed(seed: int) -> tuple[pd.DataFrame, dict] | None:
    p = SRC / f"{SRC_STEM}_seed{seed}_test_predictions.npz"
    if not p.exists():
        return None
    z = np.load(p)
    preds: dict = {}
    for k in z.files:
        tk, H, nm = k.split("|")
        preds.setdefault((tk, int(H)), {})[nm] = z[k]
    return pd.read_csv(SRC / f"{SRC_STEM}_seed{seed}_model_comparison.csv"), preds


def loss_frame(store: dict, H: int, models: list[str], preds_override: dict | None = None) -> dict[str, pd.DataFrame]:
    """모델별 (시각 × 종목) QLIKE 표. 종목마다 시각이 같아야 종목 평균 손실차가 의미를 갖는다."""
    out = {}
    for nm in models:
        cols = {}
        for (tk, h), S in store.items():
            if h != H:
                continue
            src = (preds_override or {}).get((tk, h), S["preds"])
            if nm not in src:
                continue
            T = pd.DatetimeIndex(np.asarray(S["T"]).astype("datetime64[ns]"))
            cols[tk] = pd.Series(M26.qlike_vec(S["act"].astype(float), src[nm].astype(float)), index=T)
        if cols:
            out[nm] = pd.DataFrame(cols).sort_index()
    return out


# %% [markdown]
# ## 1. 시드 견고성

# %%
def seed_section(rd: pd.DataFrame, store: dict) -> pd.DataFrame:
    emit("## 1. 시드 견고성(전 모델)")
    emit()
    runs = {0: None}
    for s in SEEDS[1:]:
        r = load_seed(s)
        if r is not None:
            runs[s] = r
    got = sorted(runs)
    emit(f"읽은 시드: {got}. 시드 0은 26번 본 실행이다.")
    emit()
    # 결정성 확인
    rows = []
    for s, r in runs.items():
        if r is None:
            continue
        _, pr = r
        for (tk, H), d in pr.items():
            for nm in DET_CHECK:
                if nm in d:
                    base = store[(tk, H)]["preds"][nm].astype(float)
                    c = float(rd[(rd["종목"] == tk) & (rd["H"] == H) & (rd["모델"] == nm)]["보정계수"].iloc[0])
                    base0 = base / np.sqrt(c) if nm == "GARCH-t" else base     # 시드 실행의 GARCH-t는 보정 전
                    rows.append({"시드": s, "모델": nm, "최대차": float(np.max(np.abs(d[nm].astype(float) - base0)))})
    det = pd.DataFrame(rows)
    if len(det):
        g = det.groupby("모델")["최대차"].max()
        emit("**결정적 알고리즘 확인**: 시드를 바꿔 다시 돌린 예측과 시드 0 예측의 최대 절대차(종목·구간 전체).")
        emit()
        emit("| 모델 | 최대 절대차 | 판정 |")
        emit("| :--- | ---: | :--- |")
        for nm, v in g.items():
            emit(f"| {nm} | {v:.2e} | {'동일(결정적)' if v < 1e-6 else '다름 — 확인 필요'} |")
        emit()
        emit("MS-GARCH·TAR-GARCH(고정 시작점의 Nelder-Mead)·KernelRidge·SVR(고정 부분표본)은 난수를 쓰지 않아 "
             "시드 실행에서 뺐다. 이 넷의 시드 표준편차는 정의상 0이다.")
        emit()
    # 무작위 모델의 시드 변동
    rows = []
    for (tk, H), S in store.items():
        a = S["act"].astype(float)
        for nm in STOCHASTIC:
            vals = [float(M26.qlike_vec(a, S["preds"][nm].astype(float)).mean())]
            for s, r in runs.items():
                if r is None:
                    continue
                d = r[1].get((tk, H), {})
                if nm in d:
                    vals.append(float(M26.qlike_vec(a, d[nm].astype(float)).mean()))
            rows.append({"종목": tk, "H": H, "모델": nm, "시드수": len(vals), "평균": np.mean(vals),
                         "표준편차": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                         "범위": np.ptp(vals)})
    sv = pd.DataFrame(rows)
    sv.to_csv(RES / f"{STEM}_seed_variability.csv", index=False)
    if sv["시드수"].max() > 1:
        emit("**무작위 모델의 시드 변동**: 종목·구간마다 시드 간 QLIKE 표준편차를 구해 중앙값을 보인다.")
        emit()
        pv = sv.pivot_table(index="모델", columns="H", values="표준편차", aggfunc="median")
        emit("| 모델 | " + " | ".join(M26.hlabel(int(h)) for h in pv.columns) + " |")
        emit("| :--- | " + " | ".join(["---:"] * len(pv.columns)) + " |")
        for nm, x in pv.iterrows():
            emit(f"| {nm} | " + " | ".join(f"{v:.4f}" for v in x.values) + " |")
        emit()
        emit(f"동률 폭 {TIE}와 비교한다. 시드 표준편차가 이보다 작으면 시드는 결론을 바꾸지 않는다. HistGBM은 표본·특성 "
             "추출을 쓰지 않는 설정이라(조기종료도 직접 감시) 시드가 결과에 영향을 주지 않는다.")
    emit()
    return sv


# %% [markdown]
# ## 2. 4시간·12시간 구분 불가 진단

# %%
def diagnose_section(rd: pd.DataFrame, store: dict, models: list[str]) -> pd.DataFrame:
    emit("## 2. 왜 긴 예측 구간에서 모델이 구분되지 않는가: 적합 실패인가, 데이터 특성인가")
    emit()
    hp = pd.read_csv(SRC / f"{SRC_STEM}_chosen_hyperparams.csv")
    rows = []
    for H in M26.HORIZONS_H:
        L = loss_frame(store, H, models)
        T_n = int(np.median([len(L[m]) for m in L]))
        # (a) 적합 진단
        lg = hp[[c for c in hp.columns if c.endswith(f"_h{H}_rounds")]] if any(c.endswith(f"_h{H}_rounds") for c in hp.columns) else None
        rounds = {c.split("_h")[0]: float(np.nanmedian(hp[c])) for c in (lg.columns if lg is not None else [])}
        ep = hp[hp.get("H", pd.Series(dtype=float)) == H] if "H" in hp.columns else pd.DataFrame()
        gru_ep = float(np.nanmedian(ep["GRU_epoch"])) if "GRU_epoch" in ep else np.nan
        lstm_ep = float(np.nanmedian(ep["LSTM_epoch"])) if "LSTM_epoch" in ep else np.nan
        # (b) 상위 모델끼리의 평균 손실차와 표준오차(종목 평균 손실차 시계열, HAC)
        mean_l = {m: L[m].mean(axis=1) for m in L}
        order = sorted(mean_l, key=lambda m: mean_l[m].mean())
        top = [m for m in order if m != "naive"][:6]
        ratios, gaps = [], []
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                d = (L[top[i]] - L[top[j]]).mean(axis=1).to_numpy()
                mu, se, _ = hac_mean_test(d)
                ratios.append(abs(mu) / se); gaps.append(abs(mu))
        # (c) 예측 유사도: 상위 모델 로그 예측의 종목별 상관 중앙값
        cors = []
        for (tk, h), S in store.items():
            if h != H:
                continue
            P = np.column_stack([np.log(S["preds"][m].astype(float)) for m in top])
            C = np.corrcoef(P.T)
            cors.append(np.median(C[np.triu_indices(len(top), 1)]))
        # (d) 예측 가능성: 최선 모델의 로그 상관·MZ R²(종목 중앙)
        best = top[0]
        lc, mz = [], []
        for (tk, h), S in store.items():
            if h != H:
                continue
            a = S["act"].astype(float); p = S["preds"][best].astype(float)
            ok = a > 0
            lc.append(np.corrcoef(np.log(a[ok]), np.log(p[ok]))[0, 1])
            mz.append(float(rd[(rd["종목"] == tk) & (rd["H"] == H) & (rd["모델"] == best)]["MZ_R2"].iloc[0]))
        cmed = rd[(rd["H"] == H) & rd["모델"].isin(M26.LOG_TARGET_MODELS)]["보정계수"].median()
        rows.append({"H": H, "평가표본(시각)": T_n, "LGBM반복": rounds.get("lgbm", np.nan),
                     "XGB반복": rounds.get("xgb", np.nan), "GRU에폭": gru_ep, "LSTM에폭": lstm_ep,
                     "로그모델보정계수": cmed, "상위6쌍 평균격차": np.median(gaps),
                     "상위6쌍 |격차|/표준오차": np.median(ratios), "상위6 예측상관": np.median(cors),
                     "최선 로그상관": np.median(lc), "최선 MZ_R2": np.median(mz), "최선": best})
    dg = pd.DataFrame(rows)
    dg.to_csv(RES / f"{STEM}_horizon_diagnosis.csv", index=False)
    emit("| 예측 구간 | 평가 시각 수 | LightGBM 반복(중앙) | XGB 반복 | GRU 에폭 | LSTM 에폭 | 로그 모델 보정계수 | "
         "상위 6개 쌍 격차 | 격차/표준오차 | 상위 6개 예측 상관 | 최선 로그 상관 | 최선 MZ R² |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, x in dg.iterrows():
        emit(f"| {M26.hlabel(int(x['H']))} | {int(x['평가표본(시각)']):,} | {x['LGBM반복']:.0f} | {x['XGB반복']:.0f} | "
             f"{x['GRU에폭']:.0f} | {x['LSTM에폭']:.0f} | {x['로그모델보정계수']:.2f} | {x['상위6쌍 평균격차']:.4f} | "
             f"{x['상위6쌍 |격차|/표준오차']:.2f} | {x['상위6 예측상관']:.3f} | {x['최선 로그상관']:.3f} | {x['최선 MZ_R2']:.3f} |")
    emit()
    emit("읽는 법: 적합 실패라면 반복 수·에폭이 1 근처에서 멈추거나(학습이 시작되지 않음) 보정계수가 비정상적으로 "
         "튄다. 데이터 특성이라면 적합 진단은 정상인데, 평가 시각이 적어 표준오차가 커지고(격차/표준오차가 2보다 "
         "작음), 상위 모델의 예측이 서로 거의 같다(예측 상관이 1에 가까움).")
    emit()
    return dg


# %% [markdown]
# ## 3. 유의성 검정

# %%
def significance_section(rd: pd.DataFrame, store: dict, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from arch.bootstrap import MCS
    emit("## 3. 유의성 검정")
    emit()
    emit("시각마다 20종목의 QLIKE 손실차를 평균한 시계열에 DM 검정(Newey-West HAC)을 한다. 종목 간 상관은 이 평균 "
         "시계열의 분산에 그대로 들어가므로 종목을 독립 반복으로 세는 문제가 없다. 다중비교는 예측 구간마다 전 모델 "
         f"쌍에 Holm 보정(유의수준 {ALPHA})을 한다. MCS는 종목 평균 손실 시계열에 블록 부트스트랩으로 돌린다"
         "(시각이 연속이라 블록 가정이 성립한다).")
    emit()
    pair_rows, mcs_rows = [], []
    for H in M26.HORIZONS_H:
        L = loss_frame(store, H, models)
        common = sorted(set.intersection(*[set(L[m].index) for m in L]))
        ML = pd.DataFrame({m: L[m].loc[common].mean(axis=1) for m in L})
        ms = list(ML.columns)
        tmp = []
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                mu, se, p = hac_mean_test((L[ms[i]].loc[common] - L[ms[j]].loc[common]).mean(axis=1).to_numpy())
                tmp.append({"H": H, "A": ms[i], "B": ms[j], "평균차(A-B)": mu, "표준오차": se, "p": p})
        t = pd.DataFrame(tmp)
        t["p_holm"] = holm(t["p"].to_numpy())
        pair_rows.append(t)
        n = len(ML)
        block = max(5, int(round(n ** (1 / 3))))
        try:
            mc = MCS(ML.to_numpy(), size=ALPHA, reps=2000, block_size=block, method="R", seed=0)
            mc.compute()
            inc = [ms[i] for i in mc.included]
            pv = mc.pvalues
            for i, nm in enumerate(ms):
                mcs_rows.append({"H": H, "모델": nm, "MCS포함": nm in inc,
                                 "MCS_p": float(pv.loc[i].iloc[0]) if i in pv.index else np.nan,
                                 "평균손실": float(ML[nm].mean()), "블록": block, "시각수": n})
        except Exception as e:
            emit(f"- {M26.hlabel(H)} MCS 실패: {type(e).__name__}: {str(e)[:100]}")
    pairs = pd.concat(pair_rows, ignore_index=True)
    mcs = pd.DataFrame(mcs_rows)
    pairs.to_csv(RES / f"{STEM}_dm_pairs.csv", index=False)
    mcs.to_csv(RES / f"{STEM}_mcs.csv", index=False)
    emit("### 3-1. 예측 구간별 MCS(모델 신뢰 집합)와 최선 대비 DM")
    emit()
    emit("MCS 포함은 \"최선이 아니라는 것을 기각할 수 없는 모델\"이다. 최선 대비 열세 p는 Holm 보정 후 값이다.")
    emit()
    for H in M26.HORIZONS_H:
        mh = mcs[mcs["H"] == H].sort_values("평균손실")
        if not len(mh):
            continue
        best = mh["모델"].iloc[0]
        ph = pairs[pairs["H"] == H]
        emit(f"#### {M26.hlabel(H)} (평가 시각 {int(mh['시각수'].iloc[0]):,}, 블록 {int(mh['블록'].iloc[0])})")
        emit()
        emit("| 모델 | 처리 방식 | 종목 평균 손실 − 최선 | MCS 포함 | 최선 대비 열세 p(Holm) |")
        emit("| :--- | :--- | ---: | :--- | ---: |")
        b0 = mh["평균손실"].iloc[0]
        for _, x in mh.iterrows():
            nm = x["모델"]
            if nm == best:
                pp = "-"
            else:
                r = ph[((ph["A"] == nm) & (ph["B"] == best)) | ((ph["A"] == best) & (ph["B"] == nm))]
                pp = f"{r['p_holm'].iloc[0]:.3g}" if len(r) else "-"
            emit(f"| {nm} | {M26.PROCESS[nm]} | {x['평균손실'] - b0:+.4f} | {'예' if x['MCS포함'] else '아니오'} | {pp} |")
        emit()
    return pairs, mcs


def regime_significance(rd: pd.DataFrame, store: dict, models: list[str]) -> pd.DataFrame:
    """사전 구간별: 시각마다 그 구간에 든 종목들의 손실차 평균 → HAC 검정, 구간 최선 대비 Holm."""
    emit("### 3-2. 사전 구간별 통계적 동률 집합")
    emit()
    emit("종목마다 사전 구간(직전 RV, 학습 구간 분위수)이 다르므로, 시각마다 그 구간에 든 종목들만으로 손실차를 "
         "평균한 시계열을 만들어 HAC 검정을 한다(해당 종목이 없는 시각은 빠진다). 구간 최선 모델 대비 열세가 "
         "Holm 보정 후 유의하지 않은 모델을 통계적 동률로 본다.")
    emit()
    rows = []
    for H in M26.HORIZONS_H:
        L = loss_frame(store, H, models)
        Q = {}
        for (tk, h), S in store.items():
            if h != H:
                continue
            T = pd.DatetimeIndex(np.asarray(S["T"]).astype("datetime64[ns]"))
            Q[tk] = pd.Series(np.digitize(S["nai"], np.quantile(S["nai_tr"], [.2, .4, .6, .8])), index=T)
        Qf = pd.DataFrame(Q).sort_index()
        for q in range(5):
            mask = (Qf == q)
            mean_l = {m: L[m].where(mask.reindex_like(L[m]).fillna(False)).stack().mean() for m in L}
            best = min((m for m in mean_l if m != "naive"), key=lambda m: mean_l[m])
            tmp = []
            for nm in L:
                if nm == best:
                    continue
                d = (L[nm] - L[best]).where(mask.reindex_like(L[nm]).fillna(False)).mean(axis=1).dropna().to_numpy()
                mu, se, p = hac_mean_test(d)
                tmp.append({"H": H, "구간": f"Q{q + 1}", "모델": nm, "최선": best, "격차": mu, "표준오차": se, "p": p})
            t = pd.DataFrame(tmp)
            t["p_holm"] = holm(t["p"].fillna(1).to_numpy())
            t["통계적동률"] = t["p_holm"] > ALPHA
            rows.append(t)
            rows.append(pd.DataFrame([{"H": H, "구간": f"Q{q + 1}", "모델": best, "최선": best, "격차": 0.0,
                                       "표준오차": 0.0, "p": 1.0, "p_holm": 1.0, "통계적동률": True}]))
    rg = pd.concat(rows, ignore_index=True)
    rg.to_csv(RES / f"{STEM}_regime_significance.csv", index=False)
    emit("| 예측 구간 | 구간 | 최선 | 통계적 동률(최선 대비 열세가 유의하지 않음) |")
    emit("| :--- | :--- | :--- | :--- |")
    for (H, q), g in rg.groupby(["H", "구간"]):
        tie = g[g["통계적동률"] & (g["모델"] != g["최선"])].sort_values("격차")
        emit(f"| {M26.hlabel(int(H))} | {q} | {g['최선'].iloc[0]} | "
             + (", ".join(f"{M26.SHORT.get(n, n)}" for n in tie["모델"]) or "-") + " |")
    emit()
    return rg


def seed_mcs_frequency(store: dict, models: list[str]) -> pd.DataFrame:
    """시드 0~4마다 무작위 모델의 예측을 그 시드 것으로 바꿔 MCS를 다시 돌리고, 모델별 포함 횟수를 센다."""
    from arch.bootstrap import MCS
    emit("### 3-3. 시드를 바꿔도 MCS 판정이 유지되는가")
    emit()
    emit("무작위 모델(트리·Nystroem·GRU·LSTM)의 예측만 시드 1~4의 것으로 바꾸고, 나머지 모델은 그대로 둔 채 같은 MCS를 "
         "다시 돌렸다. 셀은 시드 5개 중 MCS에 포함된 횟수다.")
    emit()
    seeds = {0: None}
    for sd in SEEDS[1:]:
        r = load_seed(sd)
        if r is not None:
            seeds[sd] = r[1]
    rows = []
    for H in M26.HORIZONS_H:
        for sd, pr in seeds.items():
            ov = None
            if pr is not None:
                ov = {}
                for key, S in store.items():
                    d = dict(S["preds"])
                    for nm in STOCHASTIC:
                        if nm in pr.get(key, {}):
                            d[nm] = pr[key][nm]
                    ov[key] = d
            L = loss_frame(store, H, models, ov)
            common = sorted(set.intersection(*[set(L[m].index) for m in L]))
            ML = pd.DataFrame({m: L[m].loc[common].mean(axis=1) for m in L})
            block = max(5, int(round(len(ML) ** (1 / 3))))
            mc = MCS(ML.to_numpy(), size=ALPHA, reps=2000, block_size=block, method="R", seed=0)
            mc.compute()
            inc = {list(ML.columns)[i] for i in mc.included}
            for nm in ML.columns:
                rows.append({"H": H, "시드": sd, "모델": nm, "포함": nm in inc})
    f = pd.DataFrame(rows)
    f.to_csv(RES / f"{STEM}_seed_mcs_frequency.csv", index=False)
    pv = f.groupby(["모델", "H"])["포함"].sum().unstack()
    pv = pv.loc[[m for m in M26.ALL_MODELS if m in pv.index]]
    ns = f.groupby("H")["시드"].nunique()
    emit("| 모델 | " + " | ".join(f"{M26.hlabel(int(h))}(/{ns[h]})" for h in pv.columns) + " |")
    emit("| :--- | " + " | ".join(["---:"] * len(pv.columns)) + " |")
    for nm, x in pv.iterrows():
        emit(f"| {nm} | " + " | ".join(str(int(v)) for v in x.values) + " |")
    emit()
    return f


def agreement_section(mcs: pd.DataFrame, rg: pd.DataFrame) -> None:
    emit("## 4. 실무 동률(A등급, 0.01)과 통계 판정의 일치")
    emit()
    tt = pd.read_csv(SRC / f"{SRC_STEM}_tier_table.csv")
    a = tt[(tt["기준"] == "보정후") & (tt["구간"] == "전체")]
    emit("| 예측 구간 | A등급(0.01 이내) | MCS 포함 | 둘 다 | A등급만 | MCS만 |")
    emit("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for H in M26.HORIZONS_H:
        A = set(a[(a["H"] == H) & (a["등급"] == "A")]["모델"])
        Mm = set(mcs[(mcs["H"] == H) & mcs["MCS포함"]]["모델"])
        f = lambda s_: ", ".join(M26.SHORT.get(n, n) for n in sorted(s_)) or "-"
        emit(f"| {M26.hlabel(H)} | {f(A)} | {f(Mm)} | {f(A & Mm)} | {f(A - Mm)} | {f(Mm - A)} |")
    emit()


def plot_tie_map(mcs: pd.DataFrame, rg: pd.DataFrame, models: list[str]) -> Path:
    """모델 × (예측 구간, 사전 구간) 통계적 동률 지도. 진할수록 최선과 가깝다(동률이면 칠함)."""
    cols = []
    for H in M26.HORIZONS_H:
        cols += [(H, "전체")] + [(H, f"Q{q}") for q in range(1, 6)]
    ms = [m for m in M26.ALL_MODELS if m in models and m != "naive"]
    Z = np.full((len(ms), len(cols)), np.nan)
    for j, (H, c) in enumerate(cols):
        if c == "전체":
            g = mcs[mcs["H"] == H].set_index("모델")
            for i, m in enumerate(ms):
                if m in g.index:
                    Z[i, j] = 1.0 if g.loc[m, "MCS포함"] else 0.0
        else:
            g = rg[(rg["H"] == H) & (rg["구간"] == c)].set_index("모델")
            for i, m in enumerate(ms):
                if m in g.index:
                    Z[i, j] = 1.0 if bool(g.loc[m, "통계적동률"]) else 0.0
    fig, ax = plt.subplots(figsize=(15, 6.2))
    cmap = matplotlib.colors.ListedColormap(["#f1f1ee", "#2a78d6"])
    ax.imshow(Z, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(ms)), [f"{m}  ({M26.FAMILY[m]})" for m in ms], fontsize=9)
    ax.set_xticks(range(len(cols)), [c for _, c in cols], fontsize=8)
    for k, H in enumerate(M26.HORIZONS_H):
        ax.axvline(k * 6 - 0.5, color="white", lw=4)
        ax.text(k * 6 + 2.5, -1.1, M26.hlabel(H), ha="center", fontsize=10, color="#3a3a36")
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ms), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("파란 칸 = 그 칸의 최선과 통계적으로 구분되지 않음(전체: MCS 포함, Q1~Q5: 구간 최선 대비 Holm 비유의)",
                 loc="left", fontsize=11, pad=28)
    fig.tight_layout()
    IMG.mkdir(parents=True, exist_ok=True)
    path = IMG / f"{STEM}_fig1_statistical_tie_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path



def linearity_section() -> pd.DataFrame:
    """예측 구간별 선형성: 23번의 선형성 기각(1시간 기준)이 다른 구간에서도 같은 크기인가."""
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import linear_reset
    from statsmodels.tsa.stattools import bds
    from report_header import study_universe
    emit("## 5. 예측 구간별 선형성 재검정(선형 모델 제외 결정의 근거 확인)")
    emit()
    emit("2026-09-07 가정 진단에서 선형성 검정(RESET·BDS)이 기각되어 Linear·Ridge는 비선형·커널 계열로 대체하기로 "
         "결정했다(이 분석의 순위·검정에서 제외). 그 판정은 1시간 타깃 기준이었으므로, 예측 구간마다 같은 판정이 "
         "유지되는지 다시 본다(HAR-RV도 같은 이유로 제외). "
         "표본이 수만 개면 아주 작은 비선형도 기각되므로, 종목마다 학습 표본을 최근 5,000개(긴 구간은 H시간 간격으로 겹침을 "
         "줄여 뽑은 수)로 맞추고 **비선형 항이 늘리는 설명력의 크기**를 함께 본다.")
    emit()
    tks, _ = study_universe()
    rows = []
    for tk in tks:
        D = M26.build_data(tk)
        F = M26.make_features(D)["log"]
        for H in M26.HORIZONS_H:
            HD = M26.horizon_data(D, H)
            sel = HD["tr_fit"] & (np.asarray(HD["T"].minute) == 0)
            idx = np.where(sel)[0][::max(1, H // 60)][-5000:]
            X = sm.add_constant(F[HD["j"]][idx])
            y = np.log(HD["act_d"][idx])
            ols = sm.OLS(y, X).fit()
            f = ols.fittedvalues
            r2a = sm.OLS(y, np.column_stack([X, f ** 2, f ** 3])).fit().rsquared
            rows.append({"종목": tk, "H": H, "n": len(idx), "R2": ols.rsquared, "dR2": r2a - ols.rsquared,
                         "RESET_p": float(linear_reset(ols, power=3, test_type="fitted", use_f=True).pvalue),
                         "BDS_p": float(np.atleast_1d(bds(ols.resid[-3000:], max_dim=3)[1])[0])})
    d = pd.DataFrame(rows)
    d.to_csv(RES / f"{STEM}_linearity_by_horizon.csv", index=False)
    tt = pd.read_csv(SRC / f"{SRC_STEM}_tier_table.csv")
    gap = tt[(tt["기준"] == "보정후") & (tt["구간"] == "전체")].pivot_table(index="H", columns="모델", values="격차")
    emit("| 예측 구간 | 표본(중앙) | 선형 R²(중앙) | 비선형 항 증분 R²(중앙) | RESET 기각 종목 | BDS 기각 종목 | "
         "트리 최선 격차 |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for H, g in d.groupby("H"):
        tree = min(gap.loc[H, m] for m in ("LightGBM", "XGBoost", "HistGBM", "GARCH+LightGBM"))
        emit(f"| {M26.hlabel(int(H))} | {int(g['n'].median()):,} | {g['R2'].median():.3f} | {g['dR2'].median():.4f} | "
             f"{int((g['RESET_p'] < 0.05).sum())}/{len(g)} | {int((g['BDS_p'] < 0.05).sum())}/{len(g)} | "
             f"{tree:.3f} |")
    emit()
    emit("읽는 법: 격차는 그 구간 최선 모델 대비 QLIKE 차(26번 보정후, 종목 평균)다. 선형 R²가 예측 구간과 함께 커지면 "
         "긴 구간에서 선형 근사로 충분해진다는 뜻이다(변동성을 길게 합산할수록 잡음이 평균으로 줄어든다). 비선형 항 증분이 "
         "작아도 짧은 구간에서 트리가 선형보다 크게 앞서면, 그 비선형은 RESET이 보는 거듭제곱 형태가 아니라 문턱·상호작용 "
         "형태라는 뜻이다. BDS는 잔차의 독립성 위반 전반(이분산 포함)을 잡으므로 선형성만의 검정이 아니다.")
    emit()
    return d


# %% [markdown]
# ## 실행

# %%
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-seed", action="store_true")
    a = ap.parse_args(argv)
    RES.mkdir(parents=True, exist_ok=True)
    rd, store = load_main()
    rd = rd[~rd["모델"].isin(M26.DECISION_EXCLUDED)].copy()
    for S in store.values():
        for nm in M26.DECISION_EXCLUDED:
            S["preds"].pop(nm, None)
    models = [m for m in M26.ALL_MODELS if m in set(rd["모델"])]
    tickers = sorted(rd["종목"].unique())
    emit("# 26b번: 26번 결과의 견고성·구분 불가 진단·유의성 검정")
    emit()
    emit(f"입력: `{SRC.relative_to(ROOT)}`(26번, {len(tickers)}종목 × {len(M26.HORIZONS_H)}구간 × {len(models)}모델, "
         f"{len(rd)}행). 재적합 없이 저장된 평가 예측만 쓴다. 데이터 정의와 결측 처리는 "
         "`test/research_materials/data_definition.md`, 모델 정의는 `test/research_materials/model_catalog.md`를 본다.")
    emit()
    if not a.skip_seed:
        seed_section(rd, store)
    diagnose_section(rd, store, models)
    pairs, mcs = significance_section(rd, store, models)
    rg = regime_significance(rd, store, models)
    if not a.skip_seed:
        seed_mcs_frequency(store, models)
    agreement_section(mcs, rg)
    linearity_section()
    fig = plot_tie_map(mcs, rg, models)
    emit(f"![통계적 동률 지도]({os.path.relpath(fig, RES)})")
    emit()
    (RES / f"{STEM}_report.md").write_text("\n".join(_LINES), encoding="utf-8")
    print("[26b 완료]", flush=True)


if __name__ == "__main__":
    main()
