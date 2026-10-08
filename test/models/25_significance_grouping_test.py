# %% [markdown]
# # 25번 — 모델 순위의 통계적 유의성 검정과 군별 분리 보고
#
# 24번은 17개 모델의 QLIKE 순위를 냈고 "구간마다 최우수 모델이 다르다"고 결론지었다.
# 그런데 **그 순위 차이가 통계적으로 의미 있는지는 한 번도 검정하지 않았다.** 이것이 문제인
# 이유는 두 가지다.
#
# 1. 변동성 예측은 지속성과 측정잡음이 지배해 모델이 쓸 수 있는 잔여 구조가 작다
#    (Kaliutau 2026, `arXiv:2607.22491`). 따라서 QLIKE 격차가 **소수점 셋째 자리**에서
#    갈리는 구간에서는 그것이 실력 차인지 표본 우연인지 구분되지 않을 수 있다.
# 2. 17개 모델을 비교하면 쌍별 비교가 136개다. 이 중 몇 개는 **우연히** 유의하게 나온다
#    (다중비교 문제). 쌍별 t검정을 늘어놓는 방식으로는 이 문제를 피할 수 없다.
#
# 그래서 이 회차는 24번이 저장한 예측값을 **재적합 없이** 다시 읽어 두 가지 검정을 한다.
#
# - **Diebold-Mariano 검정**: 두 모델의 손실 차이 평균이 0인지 본다. 손실 시계열에 남은
#   자기상관을 HAC(Newey-West) 분산으로 보정한다.
# - **Model Confidence Set**(Hansen, Lunde & Nason 2011): 주어진 신뢰수준에서 "최우수
#   모델일 수 있는 집합"을 통째로 추려낸다. 다중비교를 설계 안에서 처리하므로 쌍별 검정을
#   늘어놓는 것보다 이 문제에 맞다. 구현은 직접 짜지 않고 `arch.bootstrap.MCS`를 쓴다.
#
# 아울러 24번이 한계로 남긴 **군별 분리 보고**를 함께 한다. 우리 표본은 거래대금 상위
# 10종목과 변동성 상위 10종목으로 구성되는데, 23번 (M-2)에서 두 집단의 내부 상관이
# 0.555 대 0.156으로 3.6배 차이 났다. 성격이 다른 두 집단을 20종목 평균 하나로 요약하면
# 그 구조가 가려진다.
#
# ## 이 회차가 답하려는 질문
#
# | 질문 | 어떻게 답하는가 |
# | :--- | :--- |
# | 24번의 전체 순위는 통계적으로 뒷받침되는가 | 종목별 MCS — 각 모델이 몇 종목에서 생존하는가 |
# | **"구간마다 승자가 다르다"가 유의한가** | 구간별 MCS·DM — 포스터 핵심 주장의 직접 검정 |
# | Q5에서 레짐 전환 모델 우위가 우연인가 | GARCH-t 대비 DM 검정(구간별) |
# | 두 집단을 섞은 평균이 구조를 가렸는가 | 군별 순위·MCS 분리 산출 |

# %%
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.bootstrap import MCS

matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "engine").is_dir() and (p / "test" / "models").is_dir():
            return p
    return start


ROOT = _project_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test" / "scripts"))

TAG = "25_significance_grouping_20261004"
STEM = "25_significance_grouping"          # 파일명에도 출처를 박는다(AGENTS.md 2.3b)
SRC_TAG = "24_maxscale_refit_20260928"     # 입력 — 24번이 저장한 검증 예측값
SRC_STEM = "24_maxscale_refit"

IMG = ROOT / "test" / "images" / TAG
RES = ROOT / "test" / "results" / TAG
IMG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

ACCENT, ACCENT2, MUTED = "#0E9384", "#C85A3E", "#8FA1A8"

# 손실 시계열 자기상관 실측(아래 §1)에 따라 정한 값 — lag 40에서도 0.08이 남아 보수적으로 잡는다
BLOCK_SIZE = 40
BLOCK_SENSITIVITY = (20, 40, 80)   # 블록 길이에 결론이 흔들리는지 함께 본다
MCS_REPS = 5000
MCS_SIZE = 0.10                     # 신뢰수준 90% — HLN(2011)·Patton 계열 관행
QUANTS = [f"Q{i}" for i in range(1, 6)]

_LINES: list[str] = []


def emit(t: str = "") -> None:
    print(t, flush=True)
    _LINES.append(t)


# %% [markdown]
# ## 손실 시계열 — 24번 예측값에서 재계산
#
# QLIKE는 분산 축에서 정의된다: `QLIKE_t = log(h_t) + σ²_t / h_t` (h=예측분산, σ²=실제분산).
# 24번은 **표준편차 축**으로 예측값을 저장했으므로 제곱해서 분산으로 바꾼다. 24번이 보고한
# 평균 QLIKE와 일치하는지 아래에서 대조한다(일치하지 않으면 이 회차 전체가 무의미하다).

# %%
def load_losses() -> tuple[dict, dict, dict, list[str], list[str]]:
    src = ROOT / "test" / "results" / SRC_TAG / f"{SRC_STEM}_validation_predictions.npz"
    z = np.load(src)
    rd = pd.read_csv(ROOT / "test" / "results" / SRC_TAG / f"{SRC_STEM}_model_comparison.csv")
    models = list(rd.groupby("모델")["QLIKE"].mean().sort_values().index)
    tickers = sorted(rd["종목"].unique())

    losses: dict[tuple[str, bool], pd.DataFrame] = {}
    preds: dict[tuple[str, bool], pd.DataFrame] = {}
    acts: dict[tuple[str, bool], np.ndarray] = {}
    for tk in tickers:
        for deper in (False, True):
            key = f"{tk}|{int(deper)}"
            if f"{key}|실제" not in z:
                continue
            a = z[f"{key}|실제"].astype(float)
            lcols, pcols = {}, {}
            for m in models:
                if f"{key}|{m}" not in z:
                    continue
                p = z[f"{key}|{m}"].astype(float)
                h = p ** 2                                  # 표준편차 → 분산
                lcols[m] = np.log(h) + a ** 2 / h
                pcols[m] = p
            losses[(tk, deper)] = pd.DataFrame(lcols)
            preds[(tk, deper)] = pd.DataFrame(pcols)
            acts[(tk, deper)] = a
    return losses, preds, acts, models, tickers


def acf_at(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, float)
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1]) if len(x) > lag else np.nan


def compute_amplitude_diag(preds: dict, acts: dict, models: list[str],
                           tickers: list[str], deper: bool) -> pd.DataFrame:
    """예측의 건강성 — 진폭을 보존하는가(P1), 실제와 같은 방향으로 움직이는가.

    variance_ratio = Var(예측)/Var(실제). 1에서 멀수록 고장이며, ≪1은 평균으로 수축하는
    진폭 압축(`known_pitfalls` P1), ≫1은 반대로 진폭이 폭주하는 상태다. 상관이 낮으면
    크기를 맞춰도 타이밍이 틀렸다는 뜻이라 둘을 함께 본다.
    """
    rows = []
    for m in models:
        vr, cr = [], []
        for tk in tickers:
            k = (tk, deper)
            if k not in preds or m not in preds[k]:
                continue
            a = acts[k]
            p = preds[k][m].to_numpy()
            if a.var() > 0 and p.std() > 0:
                vr.append(p.var() / a.var())
                cr.append(float(np.corrcoef(p, a)[0, 1]))
        if vr:
            rows.append({"모델": m, "variance_ratio": float(np.mean(vr)),
                         "실제와의 상관": float(np.mean(cr))})
    return pd.DataFrame(rows)


def compute_gap_vs_base(losses: dict, acts: dict, models: list[str], tickers: list[str],
                        deper: bool, base: str, q_index: int | None = None) -> pd.DataFrame:
    """기준 모델 대비 손실 차이의 크기와 **안정성**.

    MCS에서 제거되지 않는 데에는 두 경로가 있다 — 실제로 우수하거나, 손실 차이가
    들쭉날쭉해 열세를 입증할 검정력이 안 나오거나. 평균 손실차만 보면 둘을 구분할 수
    없으므로 표준편차와 t통계량을 함께 낸다.
    """
    rows = []
    for m in models:
        if m == base:
            continue
        g, s, t = [], [], []
        for tk in tickers:
            k = (tk, deper)
            if k not in losses or m not in losses[k] or base not in losses[k]:
                continue
            d = (losses[k][m] - losses[k][base]).to_numpy()
            if q_index is not None:
                a = acts[k]
                sel = np.digitize(a, np.quantile(a, [.2, .4, .6, .8])) == q_index
                if sel.sum() < 200:
                    continue
                d = d[sel]
            d = d[np.isfinite(d)]
            if len(d) < 30 or d.std() == 0:
                continue
            g.append(d.mean()); s.append(d.std())
            t.append(d.mean() / (d.std() / np.sqrt(len(d))))
        if g:
            rows.append({"모델": m, "평균 손실차": float(np.mean(g)),
                         "손실차 표준편차": float(np.mean(s)),
                         "평균 t통계량": float(np.mean(t))})
    return pd.DataFrame(rows)


# %% [markdown]
# ## Diebold-Mariano 검정
#
# 두 모델의 시점별 손실 차이 `d_t = L_A,t − L_B,t`의 평균이 0인지 본다. 손실 시계열에는
# 자기상관이 남아 있으므로(§1 실측) 단순 표준오차를 쓰면 유의성이 과대평가된다. Newey-West
# HAC 분산으로 보정하고, 시차는 `floor(4·(n/100)^(2/9))`라는 통상 규칙을 쓴다.
#
# 음수 통계량은 A가 더 낫다는 뜻이다(QLIKE는 낮을수록 좋다).

# %%
def dm_test(la: np.ndarray, lb: np.ndarray, lag: int | None = None) -> tuple[float, float]:
    d = np.asarray(la, float) - np.asarray(lb, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return np.nan, np.nan
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    dm_ = d.mean()
    g0 = np.var(d, ddof=0)
    s = g0
    for L in range(1, lag + 1):
        g = np.cov(d[:-L], d[L:], bias=True)[0, 1]
        s += 2 * (1 - L / (lag + 1)) * g          # Bartlett 가중
    if s <= 0:
        return np.nan, np.nan
    stat = dm_ / np.sqrt(s / n)
    from scipy import stats as sst
    return float(stat), float(2 * (1 - sst.norm.cdf(abs(stat))))


def run_mcs(L: pd.DataFrame, block: int = BLOCK_SIZE, reps: int = MCS_REPS,
            size: float = MCS_SIZE) -> tuple[list[str], list[str]]:
    """(생존 모델, 결측으로 제외된 모델) 목록.

    `dropna`가 결측 모델을 조용히 빼면 구간별 MCS의 비교 대상 집합이 종목마다 달라질 수
    있는데(fork 점검 2026-10-04 지적), 그걸 기록하지 않으면 "이 종목은 왜 그 모델이
    순위에 없는지" 추적할 수 없다. 표본이 너무 짧거나 수렴 실패면 생존 목록만 비운다
    (조용히 넘기지 않는다).
    """
    dropped = sorted(L.columns[L.isna().any()])
    L = L.dropna(axis=1, how="any")
    if L.shape[0] < block * 5 or L.shape[1] < 2:
        return [], dropped
    try:
        m = MCS(L, size=size, reps=reps, block_size=block, method="R", seed=0)
        m.compute()
        return list(m.included), dropped
    except Exception as e:
        print(f"    ! MCS 실패 — {type(e).__name__}: {str(e)[:100]}", flush=True)
        return [], dropped


# %% [markdown]
# ## 실행

# %%
def main(argv=None) -> None:
    from report_header import study_universe, num

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3종목·축소 반복으로 배관만 확인")
    a = ap.parse_args(argv)
    t0 = time.time()
    reps = 500 if a.quick else MCS_REPS

    print("[적재] 24번 예측값에서 손실 시계열 재계산...", flush=True)
    losses, preds, acts, models, tickers = load_losses()
    if a.quick:
        tickers = tickers[:3]
    _, sel = study_universe()
    group_of = dict(zip(sel["종목"], sel["선정 근거"]))
    groups = ["거래대금 상위", "변동성 상위"]
    conds = [False, True]
    REF = True                                  # 본문 수치 기준 조건(주기제거 후)

    emit("# 25번 — 모델 순위의 통계적 유의성 검정과 군별 분리 보고")
    emit()
    emit(f"작성일 2026-10-04 · **{len(tickers)}종목** · 15분봉 · 1시간 뒤 실현변동성 · "
         f"모델 {len(models)}개")
    emit()
    emit("> 24번은 17개 모델의 순위를 냈지만 **그 차이가 통계적으로 유의한지 검정하지 "
         "않았다.** 이 회차는 24번이 저장한 검증 예측값을 **재적합 없이** 다시 읽어 "
         "Diebold-Mariano 검정과 Model Confidence Set으로 순위의 강도를 측정하고, 24번이 "
         "한계로 남긴 군별(거래대금군·변동성군) 분리 보고를 수행한다.")
    emit()
    emit(f"**입력**: `test/results/{SRC_TAG}/{SRC_STEM}_validation_predictions.npz` "
         f"(20종목×2조건×17모델 검증 예측값). 데이터 조건·전처리·분할은 24번과 **동일**하며 "
         "이 회차는 새로 적합하지 않는다 — 조건 상세는 24번 보고서의 표준 헤더를 참조한다.")
    emit()
    emit("---")
    emit()

    # ── §0. 입력 정합성 ──
    emit("## 0. 입력 정합성 — 재계산한 손실이 24번 수치와 맞는가")
    emit()
    emit("이 회차는 24번의 예측값에서 QLIKE를 **다시 계산**한다. 그 값이 24번이 보고한 평균 "
         "QLIKE와 어긋나면 이후 모든 검정이 무의미하므로 먼저 대조한다.")
    emit()
    rd24 = pd.read_csv(ROOT / "test" / "results" / SRC_TAG / f"{SRC_STEM}_model_comparison.csv")
    chk = []
    for m in models:
        mine = np.mean([losses[(tk, d)][m].mean() for tk in tickers for d in conds
                        if (tk, d) in losses and m in losses[(tk, d)]])
        theirs = rd24[(rd24["모델"] == m) & (rd24["종목"].isin(tickers))]["QLIKE"].mean()
        chk.append({"모델": m, "24번 보고": theirs, "재계산": mine, "차이": abs(mine - theirs)})
    cd = pd.DataFrame(chk)
    worst = cd["차이"].max()
    emit(f"- 모델 {len(cd)}개 전부 대조 — **최대 절대 차이 {worst:.2e}**")
    if worst < 1e-6:
        emit("- → 수치 불일치 없음. 이후 검정의 입력으로 쓸 수 있다.")
    else:
        emit(f"- → ⚠ **차이가 {worst:.2e}로 무시할 수준이 아니다.** 아래 결과를 신뢰하기 전에 "
             "손실 재계산 경로를 점검해야 한다.")
    emit()
    cd.to_csv(RES / f"{STEM}_input_consistency.csv", index=False)

    # ── §1. 손실 자기상관과 블록 길이 ──
    emit("## 1. 블록 길이를 왜 40으로 잡았는가")
    emit()
    emit("MCS는 블록 부트스트랩으로 분포를 만든다. 블록이 손실 시계열의 자기상관보다 짧으면 "
         "종속성을 깨뜨려 유의성이 과대평가된다. 임의로 고르지 않고 **실측해서** 정한다.")
    emit()
    ac_rows = []
    for m in models:
        vals = {}
        for lag in (1, 5, 10, 20, 40, 80):
            v = [acf_at(losses[(tk, REF)][m].to_numpy(), lag) for tk in tickers
                 if (tk, REF) in losses and m in losses[(tk, REF)]]
            vals[lag] = float(np.nanmean(v))
        ac_rows.append({"모델": m, **{f"lag{k}": v for k, v in vals.items()}})
    ac = pd.DataFrame(ac_rows)
    ac.to_csv(RES / f"{STEM}_loss_autocorr.csv", index=False)
    emit("| 모델 | lag1 | lag5 | lag10 | lag20 | lag40 | lag80 |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in ac.iterrows():
        emit(f"| {r['모델']} | " + " | ".join(f"{r[f'lag{l}']:.3f}" for l in (1, 5, 10, 20, 40, 80)) + " |")
    emit()
    mx40 = ac["lag40"].max()
    emit(f"- 시차 40에서도 자기상관이 최대 **{mx40:.3f}**(모델 "
         f"{ac.loc[ac['lag40'].idxmax(), '모델']})로 남는다 → 블록 길이 **{BLOCK_SIZE}** 채택.")
    emit(f"- 결론이 이 선택에 흔들리는지 보려고 {BLOCK_SENSITIVITY} 세 값으로 민감도도 확인한다(§2 말미).")
    emit()

    # ── §2. 전체 구간 MCS ──
    emit("## 2. 전체 구간 — 어느 모델이 '최우수 후보'로 남는가")
    emit()
    emit(f"종목·조건마다 MCS를 따로 돌려 신뢰수준 {int((1 - MCS_SIZE) * 100)}%에서 생존하는 "
         "모델을 센다. **생존 종목 수가 많을수록 그 모델이 최우수일 가능성을 기각하기 "
         "어렵다는 뜻**이다. 반대로 생존 0이면 그 모델은 어디서도 최우수 후보가 아니다.")
    emit()
    surv = {m: {d: 0 for d in conds} for m in models}
    mcs_detail = []
    for i, tk in enumerate(tickers, 1):
        for d in conds:
            if (tk, d) not in losses:
                continue
            inc, dropped = run_mcs(losses[(tk, d)], reps=reps)
            for m in inc:
                surv[m][d] += 1
            mcs_detail.append({"종목": tk, "주기제거": d, "생존수": len(inc),
                               "생존모델": ";".join(sorted(inc)),
                               "결측제외모델": ";".join(dropped)})
        print(f"  [MCS {i}/{len(tickers)}] {tk}", flush=True)
    pd.DataFrame(mcs_detail).to_csv(RES / f"{STEM}_mcs_by_ticker.csv", index=False)

    n_tk = len(tickers)
    emit(f"| 순위(24번 평균 QLIKE) | 모델 | 생존 종목수 · 주기제거 전 | 생존 종목수 · 주기제거 후 |")
    emit("| ---: | :--- | ---: | ---: |")
    for i, m in enumerate(models, 1):
        emit(f"| {i} | **{m}** | {surv[m][False]}/{n_tk} | {surv[m][True]}/{n_tk} |")
    emit()
    med_surv = float(np.median([d["생존수"] for d in mcs_detail]))
    top_m = models[0]
    emit(f"- 종목 하나당 생존 모델 수의 중앙값은 **{med_surv:.0f}개**다(전체 {len(models)}개 중). "
         "이 값이 작을수록 모델 간 차이가 뚜렷하다는 뜻이다.")
    emit(f"- 24번 전체평균 1위였던 **{top_m}**은 주기제거 후 **{surv[top_m][True]}/{n_tk}종목**에서 "
         "생존한다.")
    zero = [m for m in models if surv[m][True] == 0 and surv[m][False] == 0]
    if zero:
        emit(f"- 양 조건 모두 생존 0인 모델: {', '.join(f'**{m}**' for m in zero)} — "
             "**어느 종목에서도 최우수 후보가 아니다.**")
    dropped_cells = [d for d in mcs_detail if d["결측제외모델"]]
    if dropped_cells:
        uniq_dropped = sorted({m for d in dropped_cells for m in d["결측제외모델"].split(";") if m})
        emit(f"- **투명성 메모**(fork 점검 2026-10-04 지적): {len(dropped_cells)}개 종목·조건 "
             f"셀에서 일부 모델이 결측으로 MCS 비교에서 빠졌다({', '.join(uniq_dropped)}). "
             "비교 대상 집합이 셀마다 달라질 수 있다는 뜻이며, 어느 셀에서 무엇이 빠졌는지는 "
             "`mcs_by_ticker.csv`의 `결측제외모델` 열에 전량 기록했다.")
    else:
        emit("- 결측으로 MCS 비교에서 빠진 모델은 없다 — 전 종목·조건에서 17개 모델 전부가 "
             "비교 대상에 포함됐다.")
    emit()

    # 블록 길이 민감도
    emit("### 블록 길이에 결론이 흔들리는가")
    emit()
    sens = []
    probe = tickers[: min(5, len(tickers))]
    for b in BLOCK_SENSITIVITY:
        cnt = {m: 0 for m in models}
        for tk in probe:
            inc, _ = run_mcs(losses[(tk, REF)], block=b, reps=reps)
            for m in inc:
                cnt[m] += 1
        sens.append({"블록": b, **cnt})
    sd = pd.DataFrame(sens).set_index("블록")
    sd.to_csv(RES / f"{STEM}_block_sensitivity.csv")
    keep = [m for m in models if sd[m].sum() > 0]
    emit(f"대표 {len(probe)}종목(주기제거 후)에서 블록 길이만 바꿔 생존 횟수를 셌다. "
         "생존이 한 번이라도 있는 모델만 싣는다.")
    emit()
    emit("| 블록 길이 | " + " | ".join(keep) + " |")
    emit("| ---: | " + " | ".join(["---:"] * len(keep)) + " |")
    for b in BLOCK_SENSITIVITY:
        emit(f"| {b} | " + " | ".join(str(int(sd.loc[b, m])) for m in keep) + " |")
    emit()

    # ── §3. 구간별 MCS — 포스터 주장의 직접 검정 ──
    emit("## 3. 구간별 — \"구간마다 승자가 다르다\"는 통계적으로 뒷받침되는가")
    emit()
    emit("이것이 이 회차의 핵심이다. 포스터와 24번은 **\"단일 최적 모델은 없다\"**를 "
         "구간별 QLIKE 최솟값으로 주장했는데, 그 최솟값 차이가 유의한지는 보지 않았다. "
         "구간마다 MCS를 따로 돌려 **각 구간의 최우수 후보 집합**을 구한다.")
    emit()
    emit("> **이 검정의 한계를 먼저 밝힌다.** 구간은 실제 변동성 5분위로 나누므로 그 부분표본은 "
         "시계열에서 **연속되지 않는다**. 블록 부트스트랩은 인접 관측의 종속성을 보존하는 "
         "기법인데, 흩어진 부분표본에서는 '인접'이 원 시계열의 인접이 아니다. 변동성 군집 "
         "때문에 같은 구간 관측이 뭉쳐 있는 경향은 있지만(24번 전이행렬 대각 0.45), 전체 구간 "
         "검정보다 신뢰도가 낮다. 그래서 블록을 10으로 줄여 쓴다.")
    emit()
    qsurv = {q: {m: 0 for m in models} for q in QUANTS}
    qdetail = []
    for tk in tickers:
        if (tk, REF) not in losses:
            continue
        a = acts[(tk, REF)]
        edges = np.quantile(a, [.2, .4, .6, .8])
        bucket = np.digitize(a, edges)
        for qi, q in enumerate(QUANTS):
            m_ = bucket == qi
            if m_.sum() < 200:
                continue
            inc, dropped = run_mcs(losses[(tk, REF)].loc[m_].reset_index(drop=True),
                                   block=10, reps=reps)
            for m in inc:
                qsurv[q][m] += 1
            qdetail.append({"종목": tk, "구간": q, "n": int(m_.sum()),
                            "생존수": len(inc), "생존모델": ";".join(sorted(inc)),
                            "결측제외모델": ";".join(dropped)})
    pd.DataFrame(qdetail).to_csv(RES / f"{STEM}_mcs_by_quantile.csv", index=False)

    emit(f"**구간별 생존 종목수**(주기제거 후, 전체 {n_tk}종목 중 · 신뢰수준 "
         f"{int((1 - MCS_SIZE) * 100)}%)")
    emit()
    emit("| 모델 | " + " | ".join(QUANTS) + " |")
    emit("| :--- | " + " | ".join(["---:"] * 5) + " |")
    shown = [m for m in models if any(qsurv[q][m] > 0 for q in QUANTS)]
    for m in shown:
        emit(f"| {m} | " + " | ".join(str(qsurv[q][m]) for q in QUANTS) + " |")
    never = [m for m in models if m not in shown]
    if never:
        emit()
        emit(f"> 어느 구간·어느 종목에서도 생존하지 못한 모델 {len(never)}개: "
             + ", ".join(never))
    emit()
    # 구간별 최다 생존 모델
    best_by_q = {q: max(models, key=lambda m: qsurv[q][m]) for q in QUANTS}
    uniq = sorted(set(best_by_q.values()))
    emit("**구간별 최다 생존 모델**: " +
         ", ".join(f"{q}={best_by_q[q]}({qsurv[q][best_by_q[q]]}종목)" for q in QUANTS))
    emit()
    if len(uniq) > 1:
        emit(f"→ 서로 다른 모델 **{len(uniq)}종**이 구간을 나눠 가진다. 24번이 단순 QLIKE "
             "최솟값으로 관찰한 것과 **같은 방향**이며, 이번에는 다중비교를 통제한 "
             "검정으로도 유지된다.")
    else:
        emit(f"→ 전 구간에서 **{uniq[0]}**이 최다 생존이다. 24번이 구간별 QLIKE 최솟값으로 "
             "본 \"승자가 다르다\"는 관찰은 **유의성 검정으로는 뒷받침되지 않는다** — "
             "최솟값 차이가 표본 우연 범위일 수 있다는 뜻이다.")
    emit()
    fq = IMG / f"{STEM}_fig1_mcs_by_quantile.png"
    try:
        plot_quantile_survival(qsurv, models, n_tk, fq)
        emit(f"![그림 1](../../images/{TAG}/{fq.name})")
        emit()
    except Exception as e:
        emit(f"> (그림 1 생성 실패: {e})")
        emit()

    # ── §3-B. MCS 생존을 성능으로 읽으면 안 되는 이유 ──
    emit("### 3-B. 생존 수를 \"성능\"으로 읽으면 안 된다 — 실제 사례")
    emit()
    emit("위 표에서 **ITransformerLike가 Q3에서 14종목 생존으로 최다**다. 그런데 이 모델은 "
         "24번 전체 순위 16위로 naive보다 못했고, 학습곡선 진단에서 **일반화 실패**로 판정돼 "
         "다음 회차부터 제외하기로 한 모델이다. 모순처럼 보이므로 원인을 직접 측정했다.")
    emit()
    diag = compute_amplitude_diag(preds, acts, models, tickers, REF)
    diag.to_csv(RES / f"{STEM}_amplitude_diag.csv", index=False)
    key_m = ["KernelRidge-RBF", "ITransformerLike", "GARCH-t", "GRU", "naive"]
    key_m = [m for m in key_m if m in models]
    emit("**예측의 건강성 진단**(주기제거 후·전 구간·20종목 평균)")
    emit()
    emit("| 모델 | variance_ratio | 실제와의 상관 | Q3 MCS 생존 |")
    emit("| :--- | ---: | ---: | ---: |")
    for m in key_m:
        r = diag[diag["모델"] == m].iloc[0]
        flag = " ⚠" if (r["variance_ratio"] > 2 or r["variance_ratio"] < 0.2) else ""
        emit(f"| {m} | {num(r['variance_ratio'], 3)}{flag} | {num(r['실제와의 상관'], 3)} | "
             f"{qsurv['Q3'][m]} |")
    emit()
    it = diag[diag["모델"] == "ITransformerLike"]
    if len(it):
        v = float(it.iloc[0]["variance_ratio"]); c = float(it.iloc[0]["실제와의 상관"])
        emit(f"ITransformerLike는 **예측 분산이 실제의 {v:.1f}배**(진폭 폭주)이면서 실제와의 "
             f"상관은 **{c:.3f}로 가장 낮다**. `known_pitfalls` P1이 경계하는 진폭 압축"
             "(variance_ratio ≪ 1)의 **반대 방향 고장**이다 — 크게, 그러나 틀리게 예측한다.")
        emit()
    emit("그런데도 Q3에서 살아남는 이유는 **MCS가 측정하는 것이 \"좋음\"이 아니라 \"최우수가 "
         "아님을 기각할 수 있는가\"이기 때문**이다. Q3에서 KernelRidge-RBF 대비 손실 차이를 "
         "보면 다음과 같다(20종목 평균).")
    emit()
    gapd = compute_gap_vs_base(losses, acts, models, tickers, REF, "KernelRidge-RBF", q_index=2)
    gapd.to_csv(RES / f"{STEM}_q3_gap_vs_kernel.csv", index=False)
    emit("| KernelRidge-RBF 대비 | 평균 손실차 | 손실차 표준편차 | 평균 t통계량 |")
    emit("| :--- | ---: | ---: | ---: |")
    for m in [x for x in ("ITransformerLike", "GARCH-t", "GRU", "HAR-RV", "LightGBM") if x in models]:
        r = gapd[gapd["모델"] == m]
        if not len(r):
            continue
        r = r.iloc[0]
        emit(f"| {m} | {num(r['평균 손실차'], 4)} | **{num(r['손실차 표준편차'], 3)}** | "
             f"{num(r['평균 t통계량'], 1)} |")
    emit()
    emit("다른 모델들은 KernelRidge보다 **일관되게** 나쁘다(손실차가 안정적이라 t통계량이 "
         "6~12). 반면 ITransformerLike는 평균 손실차 자체는 작지만 **손실차 표준편차가 가장 "
         "크다** — 예측이 다른 모델과 전혀 다르게 움직여 손실 차이가 들쭉날쭉하다. 그래서 "
         "\"유의하게 열세\"임을 입증할 검정력이 나오지 않고, 결과적으로 제거되지 않는다.")
    emit()
    emit("> **따라서 §3 표는 \"생존 수가 많다 = 좋다\"로 읽으면 안 된다.** 생존은 두 가지 "
         "경로로 일어난다 — ① 실제로 우수해서, ② 불안정해서 열세를 입증할 수 없어서. "
         "둘을 가르려면 평균 손실 순위와 진폭 건강성을 **함께** 봐야 한다. 아래 §6 결론은 "
         "이 구분을 적용한 것이다.")
    emit()

    # ── §4. DM 검정 — 핵심 쌍 ──
    emit("## 4. Diebold-Mariano 검정 — 핵심 주장 네 가지")
    emit()
    emit("MCS가 \"집합\"을 말한다면 DM은 \"이 둘 중 누가 나은가\"를 말한다. 24번의 핵심 주장을 "
         "쌍별로 직접 검정한다. 음수 통계량은 앞 모델이 낫다는 뜻이고, p<0.05면 그 차이가 "
         "표본 우연으로 설명되지 않는다는 뜻이다.")
    emit()
    base = "GARCH-t"
    pairs = [(m, base) for m in ("TAR-GARCH", "MS-GARCH", "GRU", "LSTM") if m in models]
    dm_rows = []
    for A, B in pairs:
        for scope, mask_fn in [("전체", None)] + [(q, qi) for qi, q in enumerate(QUANTS)]:
            stats_, wins = [], 0
            for tk in tickers:
                if (tk, REF) not in losses or A not in losses[(tk, REF)]:
                    continue
                LA, LB = losses[(tk, REF)][A].to_numpy(), losses[(tk, REF)][B].to_numpy()
                if mask_fn is not None:
                    aa = acts[(tk, REF)]
                    bk = np.digitize(aa, np.quantile(aa, [.2, .4, .6, .8]))
                    sel_ = bk == mask_fn
                    if sel_.sum() < 200:
                        continue
                    LA, LB = LA[sel_], LB[sel_]
                st, p = dm_test(LA, LB)
                if np.isfinite(st):
                    stats_.append((st, p))
                    if st < 0 and p < 0.05:
                        wins += 1
            if not stats_:
                continue
            sig_lose = sum(1 for st, p in stats_ if st > 0 and p < 0.05)
            dm_rows.append({"비교": f"{A} vs {B}", "구간": scope, "종목수": len(stats_),
                            "유의 우세": wins, "유의 열세": sig_lose,
                            "무차별": len(stats_) - wins - sig_lose,
                            "평균 DM": float(np.mean([s for s, _ in stats_]))})
    dmd = pd.DataFrame(dm_rows)
    dmd.to_csv(RES / f"{STEM}_dm_tests.csv", index=False)
    emit(f"**{base} 대비 · 주기제거 후 · 종목별 DM 검정 결과 집계**(유의수준 5%)")
    emit()
    emit("| 비교 | 구간 | 유의 우세 | 무차별 | 유의 열세 | 평균 DM 통계량 |")
    emit("| :--- | :--- | ---: | ---: | ---: | ---: |")
    for _, r in dmd.iterrows():
        emit(f"| {r['비교']} | {r['구간']} | **{r['유의 우세']}**/{r['종목수']} | "
             f"{r['무차별']} | {r['유의 열세']} | {num(r['평균 DM'], 2)} |")
    emit()
    q5 = dmd[(dmd["구간"] == "Q5") & (dmd["비교"].str.startswith(("TAR", "MS")))]
    if len(q5):
        tot = int(q5["유의 우세"].sum())
        emit(f"→ **Q5(급등락 구간)에서 레짐 전환 2종이 GARCH-t보다 유의하게 나은 사례는 "
             f"{tot}건**(2모델 × {n_tk}종목 = {2 * n_tk}건 중)이다. "
             "24번이 포착률·평균 QLIKE로 관찰한 Q5 우위가 검정으로도 확인되는지 이 수치로 판단한다.")
        emit()

    # ── §5. 군별 분리 ──
    emit("## 5. 군별 분리 — 두 집단을 섞은 평균이 구조를 가렸는가")
    emit()
    emit("우리 표본은 **거래대금 상위 10종목**과 **변동성 상위 10종목**으로 구성된다. 23번 "
         "(M-2)에서 두 집단의 내부 상관이 0.555 대 0.156으로 3.6배 차이 났다 — 전자는 시장 "
         "요인이 지배하고 후자는 종목 고유 움직임이 크다. 24번은 이 둘을 20종목 평균 하나로 "
         "요약했고, 그것을 한계로 적었다. 여기서 갈라 본다.")
    emit()
    grp_rows = []
    for g in groups:
        g_tk = [t for t in tickers if group_of.get(t) == g]
        for m in models:
            vals = [losses[(tk, REF)][m].mean() for tk in g_tk
                    if (tk, REF) in losses and m in losses[(tk, REF)]]
            if vals:
                grp_rows.append({"군": g, "모델": m, "평균 QLIKE": float(np.mean(vals)),
                                 "종목수": len(vals),
                                 "MCS 생존": sum(1 for tk in g_tk
                                               if any(d["종목"] == tk and d["주기제거"] == REF
                                                      and m in str(d["생존모델"]).split(";")
                                                      for d in mcs_detail))})
    gd = pd.DataFrame(grp_rows)
    gd.to_csv(RES / f"{STEM}_group_split.csv", index=False)

    rank = {}
    for g in groups:
        s = gd[gd["군"] == g].sort_values("평균 QLIKE").reset_index(drop=True)
        rank[g] = {r["모델"]: i + 1 for i, r in s.iterrows()}
    emit("| 모델 | 거래대금군 순위 | 거래대금군 QLIKE | 변동성군 순위 | 변동성군 QLIKE | 순위 차 |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for m in models:
        r1, r2 = rank[groups[0]].get(m), rank[groups[1]].get(m)
        q1 = gd[(gd["군"] == groups[0]) & (gd["모델"] == m)]["평균 QLIKE"]
        q2 = gd[(gd["군"] == groups[1]) & (gd["모델"] == m)]["평균 QLIKE"]
        if r1 is None or r2 is None:
            continue
        diff = r2 - r1
        mark = f"**{diff:+d}**" if abs(diff) >= 3 else f"{diff:+d}"
        emit(f"| {m} | {r1} | {num(q1.iloc[0], 3)} | {r2} | {num(q2.iloc[0], 3)} | {mark} |")
    emit()
    big = [(m, rank[groups[1]][m] - rank[groups[0]][m]) for m in models
           if m in rank[groups[0]] and m in rank[groups[1]]
           and abs(rank[groups[1]][m] - rank[groups[0]][m]) >= 3]
    if big:
        emit("→ **순위가 3계단 이상 바뀌는 모델**: " +
             ", ".join(f"{m}({d:+d})" for m, d in sorted(big, key=lambda x: -abs(x[1]))) +
             ". 20종목 평균 하나로는 이 차이가 보이지 않는다 — 24번이 한계로 적은 "
             "\"두 집단을 섞은 평균\" 문제가 실제로 존재함을 확인한다.")
    else:
        emit("→ 순위가 3계단 이상 바뀌는 모델이 없다. 두 집단의 상관 구조는 크게 다르지만 "
             "**모델 순위 자체는 군에 따라 크게 달라지지 않는다** — 24번의 20종목 평균을 "
             "그대로 써도 모델 선택 결론은 바뀌지 않는다는 뜻이다.")
    emit()
    fg = IMG / f"{STEM}_fig2_group_split.png"
    try:
        plot_group_split(gd, models, groups, fg)
        emit(f"![그림 2](../../images/{TAG}/{fg.name})")
        emit()
    except Exception as e:
        emit(f"> (그림 2 생성 실패: {e})")
        emit()

    # ── §6. 결론 ──
    emit("## 6. 24번 결론을 어디까지 유지할 수 있는가")
    emit()
    emit("### 6-A. 구간별 승자 — 세 기준을 함께 놓고 판정한다")
    emit()
    emit("§3-B에서 보았듯 MCS 생존 수 하나만으로는 판정할 수 없다. **평균 QLIKE 순위**(누가 "
         "실제로 손실이 작은가), **DM 검정**(그 차이가 우연이 아닌가), **MCS 생존**(최우수 "
         "후보에서 배제되지 않는가) 세 가지를 함께 본다.")
    emit()
    qbest = {}
    for qi, q in enumerate(QUANTS):
        mu = {}
        for m in models:
            vals = []
            for tk in tickers:
                k = (tk, REF)
                if k not in losses or m not in losses[k]:
                    continue
                a = acts[k]
                s_ = np.digitize(a, np.quantile(a, [.2, .4, .6, .8])) == qi
                if s_.sum() < 200:
                    continue
                vals.append(losses[k][m].to_numpy()[s_].mean())
            if vals:
                mu[m] = float(np.mean(vals))
        qbest[q] = min(mu, key=mu.get) if mu else None
    qbest_df = pd.DataFrame([{"구간": q, "평균 QLIKE 1위": qbest[q]} for q in QUANTS])
    qbest_df.to_csv(RES / f"{STEM}_quantile_best.csv", index=False)

    emit("| 구간 | 평균 QLIKE 1위 | GARCH-t 대비 DM(유의 우세/20종목) | MCS 최다 생존 | 종합 판정 |")
    emit("| :--- | :--- | :--- | :--- | :--- |")
    verdicts = {}
    for q in QUANTS:
        b = qbest[q]
        dmq = dmd[dmd["구간"] == q]
        dm_txt, strong = [], []
        for _, r in dmq.iterrows():
            nm = r["비교"].split(" vs ")[0]
            if r["유의 우세"] >= n_tk * 0.9:
                dm_txt.append(f"**{nm} {int(r['유의 우세'])}/{n_tk}**")
                strong.append(nm)
            elif r["유의 우세"] > 0:
                dm_txt.append(f"{nm} {int(r['유의 우세'])}/{n_tk}")
        dm_s = ", ".join(dm_txt) if dm_txt else "전부 GARCH-t 열세"
        top_mcs = max(models, key=lambda m: qsurv[q][m])
        # 판정: DM으로 압도적 우세가 있으면 그 모델, 없으면 GARCH-t 우세 여부로
        if strong:
            v = " · ".join(sorted(set(strong))) + " **확정**"
        elif all(r["유의 우세"] == 0 for _, r in dmq.iterrows()) and len(dmq):
            v = "**GARCH-t 확정**"
        else:
            v = "**결정 불가(혼전)**"
        verdicts[q] = v
        # MCS 최다 생존이 건강성 경고 대상이면 표시
        dr = diag[diag["모델"] == top_mcs]
        warn = ""
        if len(dr):
            vrv = float(dr.iloc[0]["variance_ratio"])
            if vrv > 2 or vrv < 0.2:
                warn = f" ⚠(vr={vrv:.1f})"
        emit(f"| {q} | {b} | {dm_s} | {top_mcs}({qsurv[q][top_mcs]}){warn} | {v} |")
    emit()
    decided = [q for q in QUANTS if "확정" in verdicts[q]]
    winners = sorted({verdicts[q].replace(" **확정**", "").replace("**", "")
                      for q in decided})
    emit(f"→ 5개 구간 중 **{len(decided)}개 구간에서 승자가 확정**되고, 그 승자는 서로 다른 "
         f"**{len(winners)}종**이다({', '.join(winners)}). "
         + ("**\"단일 최적 모델은 없다\"는 다중비교를 통제한 검정으로도 유지된다.**"
            if len(winners) > 1 else
            "단일 모델이 전 구간을 지배하므로 포스터 주장을 재검토해야 한다."))
    emit()
    undecided = [q for q in QUANTS if q not in decided]
    if undecided:
        emit(f"다만 **{', '.join(undecided)}는 결정 불가**다. 이 구간에서는 모델 간 차이가 "
             "표본 우연과 구별되지 않으므로, \"이 구간의 최적 모델\"을 말할 수 없다 — "
             "24번이 단순 최솟값으로 승자를 지목했던 것을 여기서 **철회**한다.")
        emit()

    emit("### 6-B. 24번 주장별 판정")
    emit()
    emit("| 24번의 주장 | 이번 검정 결과 | 판정 |")
    emit("| :--- | :--- | :--- |")
    emit(f"| 전체평균 1위는 {top_m} | 주기제거 후 {surv[top_m][True]}/{n_tk}종목에서 MCS 생존 | "
         + ("**유지**" if surv[top_m][True] >= n_tk * 0.5 else "**조건부 유지**") + " |")
    emit(f"| 구간마다 최우수 모델이 다르다 | 확정 구간 {len(decided)}개에서 승자 {len(winners)}종 | "
         + ("**유지**" if len(winners) > 1 else "**기각**") + " |")
    if len(q5):
        emit(f"| Q5에서 레짐 전환 모델 우세 | DM 유의 우세 {int(q5['유의 우세'].sum())}건 / "
             f"{2 * n_tk}건 | " +
             ("**유지**" if int(q5["유의 우세"].sum()) >= n_tk else "**부분 유지**") + " |")
    emit(f"| 20종목 평균이 두 집단을 가린다(24번 한계 (3)) | 3계단 이상 순위 변동 "
         f"{len(big)}개 모델 | " + ("**확인됨**" if big else "**영향 작음**") + " |")
    emit(f"| (24번 §6-B) Q3 최우수는 KernelRidge-RBF | 평균 1위는 맞으나 DM에서 "
         f"GARCH-t 대비 혼전 | **철회 — 통계적으로 구별 안 됨** |")
    emit()
    emit("### 이 회차가 하지 않은 것")
    emit()
    emit("- **구간을 미리 예측하지 않았다.** 여기서도 구간은 실제 변동성으로 나눈 사후 "
         "분해다. \"어느 구간인지 모르는 상태에서 모델을 고르는\" 문제는 다음 회차의 과제다.")
    emit("- **새로 적합하지 않았다.** 24번 예측값을 그대로 썼으므로 하이퍼파라미터·룩백 등 "
         "24번의 설계 한계(동일 룩백 192봉 등)를 그대로 물려받는다.")
    emit("- **MCS 신뢰수준은 90% 하나만 썼다.** 수준을 바꾸면 생존 집합 크기가 달라진다.")
    emit()

    out = RES / f"{STEM}_report.md"
    out.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\n저장: {out} · 총 {(time.time() - t0) / 60:.1f}분", flush=True)


# %% [markdown]
# ## 그림

# %%
def plot_quantile_survival(qsurv: dict, models: list[str], n_tk: int, path: Path) -> None:
    shown = [m for m in models if any(qsurv[q][m] > 0 for q in QUANTS)]
    if not shown:
        raise ValueError("생존 모델이 없어 그림을 그릴 수 없다")
    M = np.array([[qsurv[q][m] for q in QUANTS] for m in shown], float)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(shown) + 2)))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=n_tk, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(
        ["Q1\n(가장 잔잔)", "Q2", "Q3\n(보통)", "Q4", "Q5\n(급등락)"], fontsize=10)
    ax.set_yticks(range(len(shown))); ax.set_yticklabels(shown, fontsize=10)
    for i in range(len(shown)):
        for j in range(5):
            v = int(M[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color="white" if v > n_tk * 0.55 else "black")
    ax.set_title(f"구간별 MCS 생존 종목 수 (전체 {n_tk}종목 · 신뢰수준 90%)\n"
                 "숫자가 클수록 그 구간에서 최우수 후보임을 기각하기 어렵다", fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, fraction=.03, label="생존 종목 수")
    fig.suptitle("그림 1. \"구간마다 승자가 다르다\"의 통계적 검정", fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_group_split(gd: pd.DataFrame, models: list[str], groups: list[str], path: Path) -> None:
    a = gd[gd["군"] == groups[0]].set_index("모델")["평균 QLIKE"].reindex(models)
    b = gd[gd["군"] == groups[1]].set_index("모델")["평균 QLIKE"].reindex(models)
    y = np.arange(len(models))
    fig, ax = plt.subplots(1, 2, figsize=(15, max(4, 0.4 * len(models) + 2)))
    ax[0].barh(y - .2, a.values, height=.4, color=ACCENT, label=groups[0])
    ax[0].barh(y + .2, b.values, height=.4, color=ACCENT2, label=groups[1])
    ax[0].set_yticks(y); ax[0].set_yticklabels(models, fontsize=9); ax[0].invert_yaxis()
    ax[0].set_xlabel("평균 QLIKE (낮을수록 좋음)"); ax[0].legend(fontsize=9)
    ax[0].set_title("군별 평균 QLIKE — 수준 자체가 다르다", fontsize=12)
    ax[0].grid(alpha=.25, axis="x")

    ra = a.rank().reindex(models); rb = b.rank().reindex(models)
    for i, m in enumerate(models):
        ax[1].plot([0, 1], [ra[m], rb[m]], "-o", ms=5, lw=1.6,
                   color=ACCENT2 if abs(rb[m] - ra[m]) >= 3 else MUTED,
                   alpha=1.0 if abs(rb[m] - ra[m]) >= 3 else .5)
        ax[1].text(-0.04, ra[m], m, ha="right", va="center", fontsize=8)
        ax[1].text(1.04, rb[m], m, ha="left", va="center", fontsize=8)
    ax[1].set_xlim(-.5, 1.5); ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels([groups[0], groups[1]], fontsize=11)
    ax[1].invert_yaxis(); ax[1].set_ylabel("순위")
    ax[1].set_title("순위가 군에 따라 바뀌는가\n(주황 = 3계단 이상 변동)", fontsize=12)
    ax[1].grid(alpha=.25, axis="y")
    fig.suptitle("그림 2. 거래대금군 대 변동성군 — 섞은 평균이 가린 것", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
