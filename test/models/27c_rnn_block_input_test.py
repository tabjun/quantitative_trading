# %% [markdown]
# # 27c번: 순환망(GRU·LSTM)에 신규 모델과 같은 블록 입력을 넣어 순차 대 병렬 구조 차이를 분리
#
# 27번에서 순환 딥러닝(GRU·LSTM)이 30분~12시간에 병렬 어텐션 계열보다 유의하게 나았다(27b D2b). 그런데 두 계열은 같은
# 15분봉 종가에서 출발해도 **모델에 넣는 모양**이 달랐다.
#
# | 항목 | 26c GRU·LSTM | 27번 신규 신경망 |
# | :--- | :--- | :--- |
# | 입력 | 15분봉 96개의 (d, \|d\|) | H분 블록 로그 RV 이력(96·96·96·60·30블록) |
# | 학습 표본 | 15분마다 겹치는 예측 시점(약 61,000행, 모든 구간) | 겹치지 않는 블록(12시간이면 약 1,270개) |
#
# 그래서 "순차 구조가 낫다"인지 "15분봉 입력·많은 표본이 낫다"인지 가를 수 없었다. 이 회차는 GRU·LSTM에 **신규 모델과 같은
# 블록 입력·같은 블록 표본·같은 보정 구간·같은 정지 확률 π**를 주고, GRU·LSTM 자체의 학습 절차(26c)는 그대로 둔다.
#
# ## 바꾸지 않는 것(26c GRU·LSTM 절차)
#
# 모델 구조(`engine/models.py`, 은닉 128), AdamW(가중감쇠 1e-4)·코사인 학습률, 학습률 후보 3개(2e-3, 5e-4, 1e-4)를 내부검증
# 손실로 고름, 최대 60에폭·조기종료 인내 10, 고른 학습률·에폭으로 학습 구간 전체 재적합, 손실은 표준화 로그 RV의 MSE.
#
# ## 바꾸는 것과 이유
#
# - **입력**: 직전 L블록의 표준화 로그 RV(L = 27번 `INPUT_BLOCKS`, 1변수). RV=0·점검 블록은 27번과 같이 직전값 유지.
#   어텐션 계열 4종 중 3종(PatchTST·Autoformer·TimeXer)과 합성곱 3종이 1변수 입력이라 1변수로 맞춘다.
# - **표본**: 블록 e(값이 있는 블록)마다 한 표본. 내부학습은 [L, k_in), 내부검증(학습률·에폭 선택, 보정)은 [k_in, k_sp),
#   전체 재적합은 [L, k_sp), 평가는 [k_sp, nb). 블록 경계·k_in·k_sp는 27번 `block_masks`와 같다.
# - **배치 크기**: 26c는 배치 2,048로 학습 행 약 61,000개를 돌아 **에폭당 약 30번** 가중치를 갱신했다. 블록 표본은 12시간에
#   약 1,270개라 같은 배치면 에폭당 1번만 갱신되어(60에폭 = 60번) 학습이 거의 되지 않는다. 그러면 입력이 아니라 학습량 차이가
#   결과를 가른다. 그래서 **에폭당 갱신 횟수를 같은 종목·구간의 26c GRU와 같게** 배치 = 2,048 × (블록 표본 수 / 26c 학습 행
#   수)로 정한다(16~2,048로 자름). 15분은 블록 하나가 15분봉 하나라 표본 수가 26c와 거의 같아 배치 2,048 그대로다.
#
# 산출: `test/results/27c_rnn_block_input_20261008/`. 시드 0~4(27번과 같은 시드 집합).

# %%
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


def _project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "engine").is_dir() and (p / "AGENTS.md").exists():
            return p
    return start


ROOT = _project_root(Path(__file__).resolve())
for _p in (ROOT, ROOT / "test" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TAG = "27c_rnn_block_input_20261008"
STEM = "27c_rnn_block_input"
RES = ROOT / "test" / "results" / TAG
IMG = ROOT / "test" / "images" / TAG


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


X = _load("x27_for27c", ROOT / "test" / "models" / "27_model_expansion_test.py")
M = X.M                                     # 26c 모듈(27번이 신규 모델 목록을 더한 상태)

BLOCK_MODELS = {"GRU-block": "GRU", "LSTM-block": "LSTM"}
for _nm, _base in BLOCK_MODELS.items():
    M.FAMILY[_nm] = "딥러닝(블록 입력)"
    M.PROCESS[_nm] = f"순차·재귀({_base}, 블록 입력)"
SEEDS = (0, 1, 2, 3, 4)
ATTN = ("PatchTST", "iTransformer", "Autoformer", "TimeXer")
CONV = ("TCN", "TimesNet", "ModernTCN")
SSM = ("S-Mamba",)
FMS = tuple(X.FM_MODELS)


def batch_for(n_block: int, n_rows26c: int) -> int:
    """26c GRU(배치 2,048, 학습 행 n_rows26c)와 에폭당 갱신 횟수가 같도록 블록 표본 n_block에 맞춘 배치."""
    return int(min(M.DL_BATCH, max(16, round(M.DL_BATCH * n_block / max(n_rows26c, 1)))))


def windows(yl: np.ndarray, idx: np.ndarray, L: int) -> np.ndarray:
    """블록 e마다 직전 L블록 yl[e-L:e](블록 e 자신은 넣지 않는다) → (len(idx), L, 1)."""
    W = np.lib.stride_tricks.sliding_window_view(yl, L)          # W[k] = yl[k:k+L]
    return np.ascontiguousarray(W[np.asarray(idx) - L])[:, :, None].astype(np.float32)


def fingerprint(quick: bool, seed: int) -> dict:
    return {"quick": bool(quick), "seed": seed, "models": list(BLOCK_MODELS),
            "data": [str(M.DATA_START), str(M.DATA_END), str(M.SPLIT), str(M.DB_PATH.name)],
            "input_blocks": {str(k): v for k, v in X.INPUT_BLOCKS.items()},
            "dl": [M.DL_HIDDEN, M.DL_MAX_EPOCHS, M.DL_PATIENCE, M.DL_BATCH, list(M.DL_LR_GRID)], "batch_rule": "2048*n_block/n_rows26c",
            "pi": "27번 π 캐시(26c 시드0 설정)"}


# %% [markdown]
# ## 작업 하나: 종목 × 구간 × 시드

# %%
def run_job(ticker: str, H: int, seed: int, quick: bool) -> dict:
    t0 = time.time()
    P = X.prepare_job(ticker, H, quick)
    B, MK, S, ymu, ysd = P["B"], P["MK"], P["S"], P["ymu"], P["ysd"]
    L = X.INPUT_BLOCKS[H] if not quick else 16
    yl = ((X.locf(B["y"]) - ymu) / ysd).astype(np.float32)
    ok = B["ok"]
    k_in, k_sp, nb = MK["k_in"], MK["k_sp"], B["nb"]
    tr = np.array([e for e in range(L, k_in) if ok[e]])
    val = np.array([e for e in range(k_in, k_sp) if ok[e]])
    full_tr = np.array([e for e in range(L, k_sp) if ok[e]])
    p_in, p_te = np.arange(k_in, k_sp), np.arange(k_sp, nb)
    n26 = int(P["HD"]["tr_in"].sum())             # 26c GRU의 내부학습 행 수(같은 종목·구간)
    bs = batch_for(len(tr), n26)
    dev = "cuda"
    fails, meta = [], {"종목": ticker, "H": H, "시드": seed, "학습표본": len(tr), "검증표본": len(val),
                       "재적합표본": len(full_tr), "26c학습행": n26, "배치": bs}
    for nm, base in BLOCK_MODELS.items():
        t1 = time.time()
        try:
            grid_lr = M.DL_LR_GRID if not quick else (M.DL_LR_GRID[0],)
            best = None
            for lr in grid_lr:
                pv, _, vl, info = M.train_dl_once(
                    base, windows(yl, tr, L), yl[tr], windows(yl, val, L), yl[val], windows(yl, p_in, L),
                    windows(yl, p_in[:1], L), dev, M.DL_MAX_EPOCHS if not quick else 3, M.DL_PATIENCE if not quick else 2,
                    bs, lr, seed=seed)
                if best is None or vl < best[2]:
                    best = (pv, vl, vl, info)
            pv, _, _, info = best
            ep = max(1, info["best_epoch"])
            _, pt, _, _ = M.train_dl_once(base, windows(yl, full_tr, L), yl[full_tr], None, None, None, windows(yl, p_te, L),
                                          dev, ep, ep + 1, bs, info["lr"], tmax=M.DL_MAX_EPOCHS if not quick else 3, seed=seed)
            full = np.full(nb, np.nan)
            full[k_in:k_sp] = pv
            full[k_sp:] = pt
            X.score_blocks(S, nm, full, MK, ymu, ysd,
                           f"블록 입력 · 배치 {bs} · lr {info['lr']:g} · {info['best_epoch']}/{info['epochs_run']}에폭 · 전체 재적합")
            meta.update({f"{nm}_lr": info["lr"], f"{nm}_epoch": info["best_epoch"], f"{nm}_epochs_run": info["epochs_run"],
                         f"{nm}_calib": S.rows[-1]["보정계수"]})
        except Exception as e:
            fails.append((nm, H, type(e).__name__, str(e)[:200]))
            traceback.print_exc()
        meta[f"{nm}_초"] = time.time() - t1
    r = X.finish_job(ticker, H, P, fails, meta, t0)
    r.pop("zinfo", None)
    return r


# %% [markdown]
# ## 자체 시험

# %%
def selftest() -> None:
    yl = np.arange(50, dtype=np.float32)
    w = windows(yl, np.array([10, 30]), 8)
    assert w.shape == (2, 8, 1) and w[0, -1, 0] == 9 and w[0, 0, 0] == 2 and w[1, -1, 0] == 29, "창 = 직전 L블록, 자신 제외"
    assert batch_for(61_508, 61_508) == 2048 and batch_for(1266, 60_850) == 43 and batch_for(100, 60_000) == 16, "배치 규칙"
    # 블록 경계·표본 구간은 27번 자체 시험(block_masks·평가 행 정렬)을 그대로 통과해야 한다
    X.selftest()
    print("[selftest 27c] 모든 시험 통과", flush=True)


# %% [markdown]
# ## 학습 실행(시드 하나, 작업 단위 저장·이어하기)

# %%
def run_seed(seed: int, quick: bool, n_tickers: int, retries: int) -> None:
    from report_header import study_universe
    tickers, _ = study_universe()
    tickers = tickers[:1] if quick else tickers[:n_tickers]
    jobs = [(tk, H) for tk in tickers for H in M.HORIZONS_H]
    stem = f"{STEM}_seed{seed}"
    part_dir = RES / "parts" / f"{stem}{'_quick' if quick else ''}"
    part_dir.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(quick, seed)
    fp_file = part_dir / "config.json"
    if fp_file.exists():
        old = json.loads(fp_file.read_text())
        if old != fp:
            raise SystemExit(f"[중단] {part_dir.name}의 저장분은 다른 설정으로 만들어졌다: "
                             f"{ {k: (old.get(k), fp.get(k)) for k in set(old) | set(fp) if old.get(k) != fp.get(k)} }")
    else:
        fp_file.write_text(json.dumps(fp, ensure_ascii=False, indent=1))
    part = lambda tk, H: part_dir / f"{tk}_{H}.pkl"
    done = {(tk, H): pickle.loads(part(tk, H).read_bytes()) for tk, H in jobs if part(tk, H).exists()}
    print(f"[시작] 27c 시드 {seed} · {len(jobs)}작업 · 저장분 {len(done)}", flush=True)
    t_start = time.time()
    for attempt in range(retries + 1):
        todo = [j for j in jobs if j not in done or done[j]["fails"]]
        if not todo:
            break
        if attempt:
            print(f"[재시도 {attempt}] {len(todo)}작업", flush=True)
        for i, (tk, H) in enumerate(todo, 1):
            try:
                r = run_job(tk, H, seed, quick)
            except Exception as e:
                traceback.print_exc()
                print(f"    ! {tk} H={H} 작업 실패 {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            done[(tk, H)] = r
            part(tk, H).write_bytes(pickle.dumps(r))
            m = r["meta"]
            print(f"  [{i}/{len(todo)}] {tk} H={H} ({r['elapsed']:.0f}s, 배치 {m['배치']}) 실패 모델 {len(r['fails'])}", flush=True)
    missing = [j for j in jobs if j not in done]
    store, store1, rows, rows1, fails, metas = {}, {}, [], [], [], []
    X.merge_jobs([done[j] for j in jobs if j in done], store, rows, rows1, store1)
    for j in jobs:
        if j in done:
            metas.append(done[j]["meta"])
            fails.extend({"종목": j[0], "모델": f[0], "H": f[1], "예외": f[2], "메시지": f[3]} for f in done[j]["fails"])
    fails.extend({"종목": tk, "모델": "(작업)", "H": H, "예외": "미완료", "메시지": ""} for tk, H in missing)
    sfx = "_quick" if quick else ""
    pd.DataFrame(rows).to_csv(RES / f"{stem}{sfx}_model_comparison.csv", index=False)
    pd.DataFrame(rows1).to_csv(RES / f"{stem}{sfx}_onepart_comparison.csv", index=False)
    pd.DataFrame(fails, columns=["종목", "모델", "H", "예외", "메시지"]).to_csv(RES / f"{stem}{sfx}_fit_failures.csv", index=False)
    pd.DataFrame(metas).to_csv(RES / f"{stem}{sfx}_meta.csv", index=False)
    M.save_npz(RES / f"{stem}{sfx}_test_predictions.npz", store, aux=True)
    print(f"[27c 시드 {seed} 완료] {(time.time() - t_start) / 60:.1f}분 · {len(rows)}행 · 실패 {len(fails)}", flush=True)


if __name__ == "__main__" and "--report" not in sys.argv:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-tickers", type=int, default=20)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    selftest()
    if not a.selftest:
        RES.mkdir(parents=True, exist_ok=True)
        run_seed(a.seed, a.quick, a.n_tickers, a.retries)


# %% [markdown]
# ## 보고서: 입력을 맞춘 뒤의 순차 대 병렬 비교
#
# 손실은 모두 시각별 QLIKE의 시드 평균(시드 0~4, 무작위 모델만 시드마다 다르고 결정적 모델은 같은 값)이다. 27번 모델의 값은
# 27번 결합 결과(`27_model_expansion_*test_predictions.npz`)에서, 블록 입력 GRU·LSTM은 이 회차의 시드별 산출물에서 읽는다.

# %%
_LINES: list[str] = []


def emit(t: str = "") -> None:
    _LINES.append(t)


def _seed_mean_losses(base: dict, seed_stores: list[dict], models: list[str]) -> dict:
    """(종목, H) → 모델 → 시각별 QLIKE 시드 평균. base는 시드 0(실제값·시각 포함), seed_stores는 시드 1~4의 예측."""
    out = {}
    for key, S in base.items():
        a = S["act"].astype(float)
        d = {}
        for nm in models:
            if nm not in S["preds"]:
                continue
            ls = [M.qlike_vec(a, S["preds"][nm].astype(float))]
            for st in seed_stores:
                p = st.get(key, {}).get("preds", {}).get(nm)
                if p is not None:
                    ls.append(M.qlike_vec(a, p.astype(float)))
            d[nm] = (np.mean(ls, axis=0), len(ls))
        out[key] = d
    return out


def _group_series(L: dict, store: dict, H: int, members: tuple) -> pd.Series:
    """구성원 손실의 단순 평균(종목마다) → 같은 시각의 종목 평균. 27b D절과 같은 정의."""
    cols = {}
    for (tk, h), d in L.items():
        if h != H:
            continue
        arr = [d[m_][0] for m_ in members if m_ in d]
        if arr:
            T = pd.DatetimeIndex(np.asarray(store[(tk, h)]["T"]).astype("datetime64[ns]"))
            cols[tk] = pd.Series(np.mean(arr, axis=0), index=T)
    return pd.DataFrame(cols).sort_index().mean(axis=1)


def write_report() -> None:
    import matplotlib.pyplot as plt
    os.environ.setdefault("RUN26B_SRC", "27")
    B26 = _load("b26_for27c", ROOT / "test" / "models" / "26b_robustness_significance_test.py")
    r27 = X.RES
    base = M._npz_to_store(r27 / "27_model_expansion_test_predictions.npz")
    s27 = [M._npz_to_store(r27 / f"27_model_expansion_seed{s}_test_predictions.npz") for s in SEEDS[1:]
           if (r27 / f"27_model_expansion_seed{s}_test_predictions.npz").exists()]
    blk = {s: M._npz_to_store(RES / f"{STEM}_seed{s}_test_predictions.npz") for s in SEEDS
           if (RES / f"{STEM}_seed{s}_test_predictions.npz").exists()}
    if 0 not in blk:
        raise RuntimeError("27c 시드 0 산출물이 없다")
    # 평가 시각·실제값이 27번과 같은지 대조
    pi_diff = 0.0
    for key, S in blk[0].items():
        if not (np.array_equal(S["T"], base[key]["T"]) and np.allclose(S["act"], base[key]["act"])):
            raise AssertionError(f"{key}: 27c와 27번의 평가 시각·실제값이 다르다")
        pi_diff = max(pi_diff, float(np.abs(S["pi"].astype(float) - base[key]["pi"].astype(float)).max()))
        base[key]["preds"].update(S["preds"])
    blk_rest = [blk[s] for s in SEEDS[1:] if s in blk]
    seeds_blk = sorted(blk)
    rows_b = pd.concat([pd.read_csv(RES / f"{STEM}_seed{s}_model_comparison.csv").assign(시드=s) for s in seeds_blk],
                       ignore_index=True)
    fails_b = pd.concat([pd.read_csv(RES / f"{STEM}_seed{s}_fit_failures.csv") for s in seeds_blk], ignore_index=True)
    meta_b = pd.concat([pd.read_csv(RES / f"{STEM}_seed{s}_meta.csv") for s in seeds_blk], ignore_index=True)
    rnn, rnnb = ("GRU", "LSTM"), tuple(BLOCK_MODELS)
    models27 = sorted({m_ for S in base.values() for m_ in S["preds"]} - set(rnnb) - {"naive"})
    L = _seed_mean_losses(base, s27, models27)
    Lb = _seed_mean_losses(base, blk_rest, list(rnnb))
    for key in L:
        L[key].update(Lb.get(key, {}))
    HS = M.HORIZONS_H
    emit("# 27c번: 순환망(GRU·LSTM)에 블록 입력을 넣은 순차 대 병렬 비교")
    emit()
    emit("## 0. 이 회차가 한 것과 이유")
    emit()
    emit("27번에서 순환 딥러닝(GRU·LSTM)이 30분~12시간에 병렬 어텐션 계열보다 유의하게 나았다(27b D2b). 두 계열은 같은 15분봉 종가에서 "
         "출발하지만 모델에 넣는 모양과 학습 표본이 달랐다. 이 회차는 GRU·LSTM에 신규 모델과 같은 블록 입력·표본을 주고 다시 비교해, "
         "순환망이 이긴 이유가 **구조**인지 **입력·표본**인지 가른다.")
    emit()
    emit("| 항목 | 26c GRU·LSTM(27번에 쓰인 값) | 블록 입력 GRU·LSTM(이 회차) | 27번 신규 신경망 |")
    emit("| :--- | :--- | :--- | :--- |")
    emit("| 원천 데이터 | 15분봉 종가 | 같음 | 같음 |")
    emit("| 입력 | 15분봉 96개의 (d, \\|d\\|) | 직전 L블록 표준화 로그 RV(1변수), L=96·96·96·60·30 | 같은 블록 이력(iTransformer·S-Mamba는 +블록 수익률) |")
    emit("| 학습 표본 | 15분마다 겹치는 예측 시점 | 겹치지 않는 블록 | 겹치지 않는 블록 |")
    emit("| 학습률·에폭 선택 | 내부검증 구간(학습률 3개, 최대 60에폭, 인내 10) | 같음(블록 단위 내부검증 구간) | 학습 구간 끝 15%로 조기종료 |")
    emit("| 보정·정지 확률 π | 26c 내부검증 정시, 26c π | 27번 블록 경계 정시, 27번 π | 27번 블록 경계 정시, 27번 π |")
    emit("| 배치 | 2,048 | 2,048 × (블록 표본 수 / 26c 학습 행 수) | 라이브러리·저자 설정 |")
    emit()
    bt = meta_b[meta_b["시드"] == 0].groupby("H")[["학습표본", "26c학습행", "배치"]].median()
    emit("**배치 규칙의 이유**: 26c GRU는 학습 행 약 6만 개를 배치 2,048로 돌아 에폭당 약 30번 가중치를 갱신했다. 블록 표본은 12시간에 "
         "약 1,270개라 같은 배치면 에폭당 1번만 갱신되어, 입력이 아니라 학습량 때문에 결과가 갈린다. 그래서 같은 종목·구간의 26c와 "
         "에폭당 갱신 횟수가 같도록 배치만 줄였다. 시드 0의 종목 중앙값은 다음과 같다.")
    emit()
    emit("| 구간 | 블록 학습 표본 | 26c 학습 행 | 배치 |")
    emit("| :--- | ---: | ---: | ---: |")
    for H in HS:
        emit(f"| {M.hlabel(H)} | {bt.loc[H, '학습표본']:,.0f} | {bt.loc[H, '26c학습행']:,.0f} | {bt.loc[H, '배치']:,.0f} |")
    emit()
    emit("**하지 않은 것**: 신규 모델에 15분봉을 넣는 반대 방향 통일은 하지 않았다. 신규 모델은 '같은 시계열의 다음 값'을 예측하는 "
         "라이브러리 구조라, 15분봉을 넣으려면 15분 값 여러 개를 예측해 합치는 다단계 예측으로 바꾸거나(출력 형태라는 새 차이가 생김) "
         "학습 절차를 고쳐야 한다(표준 사용법에서 벗어남). GRU는 어떤 수열이든 받는 범용 순차 회귀라 입력만 바꿔도 절차가 그대로다.")
    emit()
    # ---- 1. 산출 점검
    emit("## 1. 결과 산출 점검")
    emit()
    nan_n = int(rows_b["QLIKE"].isna().sum())
    dup_n = int(rows_b.duplicated(["시드", "종목", "H", "모델"]).sum())
    emit(f"- 읽은 시드: {seeds_blk}. 시드마다 20종목 × 5구간 × 2모델 = 200행이 기대값이다. 실제 행 수: "
         + ", ".join(f"시드 {s} {int((rows_b['시드'] == s).sum())}행" for s in seeds_blk) + ".")
    emit(f"- QLIKE 결측 {nan_n}행, (시드, 종목, 구간, 모델) 중복 {dup_n}행, 적합 실패 {len(fails_b)}건.")
    emit(f"- 평가 시각과 실제값은 27번 결합 결과와 100칸 모두 같다(합치는 단계에서 대조, 다르면 멈춘다).")
    emit(f"- 정지 확률 π: 27번 결합 결과(26c GRU·LSTM이 쓴 π)와의 최대 절대차 {pi_diff:.2e}. 0이면 두 GRU의 차이에 π는 관여하지 않는다. "
         "15분은 블록 경계가 모든 15분 시점이라 보정 구간도 26c와 같으므로, 15분의 GRU 대 블록 입력 GRU 차이는 입력 표현 차이만 남는다.")
    cal = rows_b["보정계수"]
    emit(f"- 보정계수(두 부분 모형의 크기 배율) 범위 {cal.min():.2f}~{cal.max():.2f}, 중앙 {cal.median():.2f}. "
         f"27번 전체 범위(0.36~9.65) 안이면 정상으로 본다.")
    emit()
    # ---- 2. 격차 표
    emit("## 2. 구간별 손실: 27번 최선 대비 격차")
    emit()
    show = ["GRU", "LSTM", "GRU-block", "LSTM-block", "PatchTST", "TimesFM"]
    cell = []
    for (tk, H), d in L.items():
        for m_, (v, ns) in d.items():
            cell.append({"종목": tk, "H": H, "모델": m_, "QLIKE": float(v.mean()), "시드수": ns})
    cell = pd.DataFrame(cell)
    mean_ = cell.groupby(["H", "모델"])["QLIKE"].mean().unstack()
    best27 = mean_[models27].idxmin(axis=1)
    gap = {}
    for H in HS:
        b = best27[H]
        cH = cell[cell["H"] == H].pivot(index="종목", columns="모델", values="QLIKE")
        gap[H] = (cH.sub(cH[b], axis=0)).mean()
    gap = pd.DataFrame(gap)
    gap.to_csv(RES / f"{STEM}_gap_table.csv")
    emit("| 모델 | 입력 | " + " | ".join(M.hlabel(H) for H in HS) + " |")
    emit("| :--- | :--- | " + " | ".join(["---:"] * len(HS)) + " |")
    inp = {"GRU": "15분봉", "LSTM": "15분봉", "GRU-block": "블록", "LSTM-block": "블록", "PatchTST": "블록", "TimesFM": "블록(512)"}
    for m_ in show:
        emit(f"| {m_} | {inp[m_]} | " + " | ".join(f"{gap.loc[m_, H]:+.4f}" for H in HS) + " |")
    emit("| (27번 최선) | | " + " | ".join(best27[H] for H in HS) + " |")
    emit()
    emit("**읽는 법**: 셀은 종목마다 (그 모델 − 그 구간 27번 최선 모델)의 시드 평균 QLIKE 차를 20종목 평균한 값이다(0에 가까울수록 "
         "최선에 가깝고, 0.01 이하는 27번의 A등급 기준). PatchTST는 27번 어텐션 계열 최선, TimesFM은 파운데이션 계열 최선이다.")
    emit()
    ins = []
    for H in HS:
        g0, g1 = gap.loc["GRU", H], gap.loc["GRU-block", H]
        ins.append(f"{M.hlabel(H)} GRU {g0:+.4f} → 블록 {g1:+.4f}")
    emit("**해석**: 같은 GRU의 입력만 바꿨을 때 격차 변화는 " + ", ".join(ins) + "이다. 격차가 커지면 15분봉 입력·많은 표본이 GRU에 "
         "유리했던 것이고, 비슷하면 입력이 결과를 가르지 않았다는 뜻이다. 통계적 판정은 3절의 검정으로 한다.")
    emit()
    # ---- 3. 검정
    emit("## 3. 검정: 구조 효과와 입력 효과")
    emit()
    emit("**검정 설계**")
    emit()
    emit("| 검정 | 비교 집단(A − B) | H0 | H1 | H0 기각의 의미 |")
    emit("| :--- | :--- | :--- | :--- | :--- |")
    emit("| S1 구조 효과(같은 입력) | 순환(블록 입력: GRU-block·LSTM-block) − 병렬 어텐션(PatchTST·iTransformer·Autoformer·TimeXer) | 두 계열의 기대 손실이 같다 | 다르다 | 입력·표본을 맞춰도 차이가 남는다 → 구조(순차 대 병렬) 차이다 |")
    emit("| S2 구조 효과(합성곱) | 순환(블록 입력) − 병렬 합성곱(TCN·TimesNet·ModernTCN) | 같다 | 다르다 | 같은 입력에서 순환과 합성곱 구조가 다르다 |")
    emit("| S3 구조 효과(상태공간) | 순환(블록 입력) − S-Mamba | 같다 | 다르다 | 같은 입력에서 순환과 선택적 상태공간이 다르다 |")
    emit("| I1 입력 효과(같은 구조) | 순환(15분봉 입력: GRU·LSTM) − 순환(블록 입력) | 같다 | 다르다 | 같은 구조에서 입력·표본만 바꿔도 손실이 달라진다 → 27번 차이에 입력 효과가 섞여 있었다 |")
    emit("| R1 참고(27b 재현) | 순환(15분봉 입력) − 병렬 어텐션 | 같다 | 다르다 | 27b D2b의 결과(입력이 다른 상태의 비교) |")
    emit("| R2 참고 | 순환(블록 입력) − 파운데이션(zero-shot) | 같다 | 다르다 | 입력 길이가 다르다(파운데이션은 512블록 이상)는 점이 남은 비교 |")
    emit()
    emit("계열 손실은 시각마다 구성원 QLIKE(시드 평균)를 단순 평균한 값을 종목에 걸쳐 평균한 것이다(27b D절과 같은 정의, 대표 모델을 고르지 "
         "않아 선택 편향이 없다). 통계량은 대응 차이 d_t의 HAC t(Newey-West, lag=⌊4(n/100)^(2/9)⌋)이고, 구간마다 위 6개 검정에 Holm 보정, "
         "유의수준 0.05다. 평균차가 음수면 A가 낫다.")
    emit()
    groups = {"순환(블록)": rnnb, "순환(15분봉)": rnn, "어텐션": ATTN, "합성곱": CONV, "상태공간": SSM, "파운데이션": FMS}
    tests = [("S1", "순환(블록)", "어텐션"), ("S2", "순환(블록)", "합성곱"), ("S3", "순환(블록)", "상태공간"),
             ("I1", "순환(15분봉)", "순환(블록)"), ("R1", "순환(15분봉)", "어텐션"), ("R2", "순환(블록)", "파운데이션")]
    mtests = [("M1", "GRU", "GRU-block"), ("M2", "LSTM", "LSTM-block"), ("M3", "GRU-block", "PatchTST"),
              ("M4", "LSTM-block", "PatchTST")]
    rows = []
    for H in HS:
        gs = pd.DataFrame({g: _group_series(L, base, H, mem) for g, mem in groups.items()}).dropna()
        ms = pd.DataFrame({m_: _group_series(L, base, H, (m_,)) for m_ in ("GRU", "LSTM", "GRU-block", "LSTM-block", "PatchTST")}).dropna()
        for kind, lst, frame in (("계열", tests, gs), ("모델", mtests, ms)):
            part = []
            for tid, A, Bn in lst:
                mu, se, p = B26.hac_mean_test((frame[A] - frame[Bn]).to_numpy())
                part.append({"H": H, "종류": kind, "검정": tid, "A": A, "B": Bn, "평균차(A-B)": mu, "표준오차": se, "p": p,
                             "시각수": len(frame)})
            ph = B26.holm(np.array([r["p"] for r in part]))
            for r, q in zip(part, ph):
                r["p_holm"] = q
            rows.extend(part)
    T = pd.DataFrame(rows)
    T.to_csv(RES / f"{STEM}_dm_tests.csv", index=False)

    def verdict(r) -> str:
        if r["p_holm"] >= 0.05:
            return "구분 안 됨"
        return f"{r['A'] if r['평균차(A-B)'] < 0 else r['B']} 우세"

    emit("### 3-1. 계열 검정(Holm 보정 p, 구간마다 6개 검정)")
    emit()
    emit("| 검정 | 비교(A − B) | " + " | ".join(M.hlabel(H) for H in HS) + " |")
    emit("| :--- | :--- | " + " | ".join([":---"] * len(HS)) + " |")
    for tid, A, Bn in tests:
        cells = []
        for H in HS:
            r = T[(T["H"] == H) & (T["검정"] == tid)].iloc[0]
            cells.append(f"{r['평균차(A-B)']:+.4f} (p={r['p_holm']:.2g}, {verdict(r)})")
        emit(f"| {tid} | {A} − {Bn} | " + " | ".join(cells) + " |")
    emit()
    nT = T[T["종류"] == "계열"].groupby("H")["시각수"].first()
    emit("평가 시각 수: " + ", ".join(f"{M.hlabel(H)} {nT[H]:,}" for H in HS) + ".")
    emit()
    emit("### 3-2. 모델 검정(Holm 보정 p, 구간마다 4개 검정)")
    emit()
    emit("| 검정 | 비교(A − B) | " + " | ".join(M.hlabel(H) for H in HS) + " |")
    emit("| :--- | :--- | " + " | ".join([":---"] * len(HS)) + " |")
    for tid, A, Bn in mtests:
        cells = []
        for H in HS:
            r = T[(T["H"] == H) & (T["검정"] == tid)].iloc[0]
            cells.append(f"{r['평균차(A-B)']:+.4f} (p={r['p_holm']:.2g}, {verdict(r)})")
        emit(f"| {tid} | {A} − {Bn} | " + " | ".join(cells) + " |")
    emit()
    # ---- 4. 구간별 판정
    emit("## 4. 구간별 판정: 27번의 '순환 딥러닝 > 어텐션'은 구조 때문인가")
    emit()
    emit("| 구간 | R1 27번 비교(입력 다름) | S1 같은 입력(계열 평균) | M3 같은 입력(GRU 대 PatchTST) | I1 입력 효과 | 판정 |")
    emit("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for H in HS:
        g = {t: T[(T["H"] == H) & (T["검정"] == t)].iloc[0] for t in ("R1", "S1", "I1", "M3")}
        r1, s1, i1, m3 = (verdict(g[t]) for t in ("R1", "S1", "I1", "M3"))
        s1_rnn = s1.startswith("순환")
        r1_rnn = r1.startswith("순환")
        i1_sig = i1 != "구분 안 됨"
        if r1_rnn and s1_rnn:
            v = ("구조 효과: 입력·표본을 맞춰도 순환 계열이 낫다" + (" (입력 효과도 함께 있음)" if i1_sig else "")
                 + ("" if m3 != "구분 안 됨" else ". 단 최선끼리(GRU 대 PatchTST)는 구분되지 않는다"))
        elif r1_rnn and not s1_rnn:
            v = "입력·표본 효과: 맞추면 순환의 우위가 사라진다" if i1_sig else "판정 보류: 같은 입력에서 우위가 사라졌지만 입력 효과도 유의하지 않다"
        elif not r1_rnn and s1_rnn:
            v = "같은 입력에서만 순환이 낫다(15분봉 입력이 순환에 불리했음)"
        else:
            v = "해당 없음(27번에서도 순환 우위가 없던 구간)" if r1 == "구분 안 됨" else f"27번에서 {r1}"
        emit(f"| {M.hlabel(H)} | {r1} | {s1} | {m3} | {i1} | {v} |")
    emit()
    emit("**읽는 법**: R1이 27번의 결론이고, S1이 입력·표본을 맞춘 뒤의 같은 비교다. I1은 GRU·LSTM의 입력만 바꿨을 때의 차이다. "
         "R1과 S1이 같은 방향으로 유의하면 순환 구조 자체가 낫다는 근거가 되고(계열 평균, 곧 전형적인 모델끼리의 비교), M3는 각 계열의 대표 모델끼리 같은 입력에서 비교한 결과다. 계열 평균의 우위가 약한 구성원(예: iTransformer·Autoformer) 때문이면 M3가 구분되지 않는다. R1은 유의한데 S1이 구분되지 않고 I1이 유의하면 27번 차이는 "
         "입력·표본에서 왔다고 본다. H0를 기각하지 못한 것은 같다는 증명이 아니라 차이의 증거가 부족하다는 뜻이다.")
    emit()
    # ---- 5. 시드 변동
    emit("## 5. 시드 변동")
    emit()
    sd = rows_b.groupby(["종목", "H", "모델"])["QLIKE"].std().groupby(["H", "모델"]).median().unstack()
    emit("| 모델 | " + " | ".join(M.hlabel(H) for H in HS) + " |")
    emit("| :--- | " + " | ".join(["---:"] * len(HS)) + " |")
    for m_ in rnnb:
        emit(f"| {m_} | " + " | ".join(f"{sd.loc[H, m_]:.4f}" for H in HS) + " |")
    emit()
    emit(f"**읽는 법**: 종목·구간마다 시드 {len(seeds_blk)}개의 QLIKE 표준편차를 구해 구간별 중앙값을 보였다. 0.01(27번 동률 폭)보다 작으면 "
         "시드가 위 판정을 바꾸지 않는다.")
    emit()
    # ---- 그림
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    sty = {"GRU": ("#eda100", "-", "o"), "LSTM": ("#eda100", "--", "s"), "GRU-block": ("#8a5a00", "-", "o"),
           "LSTM-block": ("#8a5a00", "--", "s"), "PatchTST": ("#4a3aa7", "-", "X"), "TimesFM": ("#6b6b66", "-", "*")}
    lab = {"GRU": "GRU(15분봉 입력)", "LSTM": "LSTM(15분봉 입력)", "GRU-block": "GRU(블록 입력)", "LSTM-block": "LSTM(블록 입력)",
           "PatchTST": "PatchTST(어텐션 최선)", "TimesFM": "TimesFM(파운데이션 최선)"}
    xs = np.arange(len(HS))
    for m_ in show:
        c, ls, mk = sty[m_]
        ax.plot(xs, [gap.loc[m_, H] for H in HS], color=c, ls=ls, marker=mk, lw=1.6, ms=5, label=lab[m_])
    ax.axhline(0.01, color="#8a8a85", lw=0.8, ls=":")
    ax.text(len(HS) - 1, 0.011, "A등급 기준 0.01", ha="right", va="bottom", fontsize=7.5, color="#8a8a85")
    ax.set_xticks(xs, [M.hlabel(H) for H in HS])
    ax.set_ylabel("27번 최선 대비 QLIKE 격차(시드 평균)")
    ax.grid(axis="y", color="#e6e6e3", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("입력을 바꾼 GRU·LSTM과 신규 모델의 격차", fontsize=10, loc="left")
    hd, lb = ax.get_legend_handles_labels()
    fig.legend(hd, lb, fontsize=8, frameon=False, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    IMG.mkdir(parents=True, exist_ok=True)
    fp = IMG / f"{STEM}_fig1_gap_by_input.png"
    fig.savefig(fp, dpi=140, bbox_inches="tight")
    plt.close(fig)
    emit("## 6. 그림")
    emit()
    emit(f"![입력별 격차]({os.path.relpath(fp, RES)})")
    emit()
    emit("**읽는 법**: 가로축은 예측 구간, 세로축은 2절 표의 격차다(낮을수록 좋음, 점선 아래는 A등급). 같은 색의 진한 선이 블록 입력, "
         "연한 선이 15분봉 입력이다. 두 선의 간격이 입력 효과이고, 진한 선과 PatchTST 선의 간격이 같은 입력에서의 구조 차이다.")
    emit()
    (RES / f"{STEM}_report.md").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"[보고서] {RES / f'{STEM}_report.md'}", flush=True)


if __name__ == "__main__" and "--report" in sys.argv:
    selftest()
    write_report()
