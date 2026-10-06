"""27번 파운데이션 모델 러너(zero-shot, 단일 모델·단일 venv).

입력 디렉터리의 `{종목}_{H}.npz`(series: 표준화한 로그 RV 블록 시계열 LOCF, orig: 예측 시점 블록 인덱스, ds0: 첫 블록
시각, H: 블록 길이(분))를 읽어, 각 예측 시점 k마다 직전 블록 `series[k-ctx:k]`만으로 다음 한 블록을 예측한다. 재학습은 없다.
출력은 `{출력 디렉터리}/{종목}_{H}.npz`(orig, pred, 표준화 단위).

모델마다 필요한 패키지가 달라 서로 다른 venv에서 이 스크립트를 따로 부른다:
  Chronos-Bolt·TimesFM·TTM : 메인 `.venv`
  Moirai-2                 : `.venvs/moirai_py312_20261004_192438`
  Sundial·Time-MoE         : `.venvs/legacy_hf_py312_20261004`
  Lag-Llama                : `.venvs/lagllama_py312_20261004`

점예측 규칙(로그 RV의 조건부 중심): 분위수·표본을 내는 모델은 **중앙값**을 쓴다(Chronos-Bolt q0.5, Moirai-2 q0.5,
Sundial·Lag-Llama 표본 중앙값). TimesFM은 점예측 헤드, TTM은 첫 예측 스텝, Time-MoE는 점예측이다. 표본 기반 모델(Sundial,
Lag-Llama)은 `torch.manual_seed(0)`로 고정한다.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
CTX = {"Chronos-Bolt": 512, "TimesFM": 512, "TTM": 512, "Moirai-2": 512, "Sundial": 512, "Time-MoE": 512, "Lag-Llama": 1124}
FREQ = {15: "15min", 30: "30min", 60: "h", 240: "4h", 720: "12h"}


def build(model: str, device: str):
    import torch
    torch.manual_seed(0)
    if model == "Chronos-Bolt":
        from chronos import BaseChronosPipeline
        p = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=device, torch_dtype=torch.float32)

        def f(x, meta):
            q, _ = p.predict_quantiles([torch.tensor(r) for r in x], prediction_length=1, quantile_levels=[0.5])
            return q[:, 0, 0].float().cpu().numpy()
        return f
    if model == "TimesFM":
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=8, normalize_inputs=True, per_core_batch_size=256,
                                         use_continuous_quantile_head=False, force_flip_invariance=False,
                                         infer_is_positive=False, fix_quantile_crossing=False))

        def f(x, meta):
            pt, _ = m.forecast(horizon=1, inputs=[r for r in x])
            return np.asarray(pt)[:, 0]
        return f
    if model == "TTM":
        from tsfm_public import TinyTimeMixerForPrediction
        m = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r2").to(device).eval()

        def f(x, meta):
            with torch.no_grad():
                o = m(past_values=torch.tensor(x[:, :, None]).to(device))
            return o.prediction_outputs[:, 0, 0].float().cpu().numpy()
        return f
    if model == "Moirai-2":
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        mod = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
        m = Moirai2Forecast(module=mod, prediction_length=1, context_length=512, target_dim=1,
                            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).to(device)

        def f(x, meta):
            out = m.predict([r[:, None] for r in x])         # (B, 9 분위수, 1)
            return np.asarray(out)[:, 4, 0]                   # 분위수 0.1..0.9의 가운데 = 중앙값
        return f
    if model in ("Sundial", "Time-MoE"):
        from transformers import AutoModelForCausalLM
        name = "thuml/sundial-base-128m" if model == "Sundial" else "Maple728/TimeMoE-50M"
        m = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True).to(device).eval()

        def f(x, meta):
            xt = torch.tensor(x).to(device)
            with torch.no_grad():
                if model == "Sundial":
                    o = m.generate(xt, max_new_tokens=1, num_samples=50)            # (B, 50, 1)
                    return o[:, :, 0].median(dim=1).values.float().cpu().numpy()
                mu, sd = xt.mean(dim=1, keepdim=True), xt.std(dim=1, keepdim=True).clamp_min(1e-6)
                o = m.generate((xt - mu) / sd, max_new_tokens=1)                    # (B, L+1)
                return (o[:, -1:] * sd + mu)[:, 0].float().cpu().numpy()
        return f
    if model == "Lag-Llama":
        import pandas as pd
        sys.path.insert(0, str(ROOT / "third_party" / "lag-llama"))
        _tl = torch.load
        torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})   # 공식 체크포인트(신뢰 출처), gluonts 객체 포함
        from lag_llama.gluon.estimator import LagLlamaEstimator
        from gluonts.dataset.common import ListDataset
        ck = str(ROOT / "third_party" / "lag-llama" / "lag-llama.ckpt")
        a = torch.load(ck, map_location=device)["hyper_parameters"]["model_kwargs"]
        est = LagLlamaEstimator(ckpt_path=ck, prediction_length=1, context_length=32, input_size=a["input_size"],
                                n_layer=a["n_layer"], n_embd_per_head=a["n_embd_per_head"], n_head=a["n_head"],
                                scaling=a["scaling"], time_feat=a["time_feat"], batch_size=256, num_parallel_samples=100,
                                device=torch.device(device))
        pred = est.create_predictor(est.create_transformation(), est.create_lightning_module())

        def f(x, meta):
            H = meta["H"]
            ds = ListDataset([{"start": pd.Timestamp(meta["ds0"]) + pd.Timedelta(minutes=int(k) * H) - pd.Timedelta(minutes=H * len(r)),
                               "target": r} for r, k in zip(x, meta["k"])], freq=FREQ[H])
            return np.array([np.median(fc.samples[:, 0]) for fc in pred.predict(ds)], dtype=np.float32)
        return f
    raise KeyError(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CTX))
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="디버깅용: 파일당 예측 시점 수 제한")
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.batch <= 1:
        raise SystemExit("배치가 1 이하다. model_catalog.md 6-1절 점검표를 확인하라")
    f = build(a.model, device)
    L = CTX[a.model]
    print(f"[{a.model}] 문맥 {L} · 외부 배치 {a.batch}" + (" · TimesFM per_core_batch_size 256" if a.model == "TimesFM" else ""), flush=True)
    t0 = time.time()
    files = sorted(Path(a.inputs).glob("*.npz"))
    for i, fp in enumerate(files, 1):
        dst = out / fp.name
        if dst.exists():
            continue
        z = np.load(fp, allow_pickle=True)
        s, orig, H, ds0 = z["series"].astype(np.float32), z["orig"].astype(np.int64), int(z["H"]), str(z["ds0"])
        if a.limit:
            orig = orig[:a.limit]
        if orig.min() < L:
            raise AssertionError(f"{fp.name}: 이력이 {L}블록 미만인 예측 시점이 있다(최소 {orig.min()})")
        pred = np.empty(len(orig), np.float32)
        b, bs = 0, a.batch
        while b < len(orig):
            kk = orig[b:b + bs]
            x = np.stack([s[k - L:k] for k in kk]).astype(np.float32)
            try:
                p = np.asarray(f(x, {"H": H, "ds0": ds0, "k": kk}), np.float32)
            except torch.OutOfMemoryError:
                # GPU 메모리 부족: 캐시를 비우고 배치를 절반으로 줄여 같은 구간을 다시 예측한다(결과는 배치 크기와 무관)
                torch.cuda.empty_cache()
                if bs <= 8:
                    raise
                bs //= 2
                print(f"    메모리 부족 → 배치 {bs}로 축소", flush=True)
                continue
            if p.shape != (len(kk),) or not np.all(np.isfinite(p)):
                raise FloatingPointError(f"{fp.name}: 예측이 비정상 {p.shape}")
            pred[b:b + len(kk)] = p
            b += len(kk)
        np.savez_compressed(dst, orig=orig, pred=pred)
        print(f"  [{a.model} {i}/{len(files)}] {fp.name} {len(orig)}건 ({time.time() - t0:.0f}s)", flush=True)
    print(f"[{a.model} 완료] {(time.time() - t0) / 60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
