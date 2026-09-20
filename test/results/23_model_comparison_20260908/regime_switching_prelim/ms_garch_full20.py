"""GARCH-t 대 레짐전환 GARCH(MS-GARCH)를 top20 전체·2조건(주기제거 전/후)으로 비교한다.
BTC 1종목 첫 확인에서 Q5(최고변동구간)에서만 MS-GARCH가 우세한 신호를 봤는데,
이게 BTC만의 우연인지 20종목 전체에서 재현되는지 확인하는 용도(8종목 편향 사건과
같은 원칙 — 표본 하나로 결론 내지 않는다)."""
import warnings; warnings.filterwarnings("ignore")
import sys, time, importlib.util
import numpy as np
import pandas as pd

sys.path.insert(0, "/tmp/claude-1002/-home-std-jun99120-personal-ai-project-tools-vscode/fc52cb5c-495a-4a72-a905-2a961a95d970/scratchpad/poster")
from ms_garch import fit_ms_garch

sys.path.insert(0, "test/scripts")
from report_header import study_universe

spec = importlib.util.spec_from_file_location(
    "m23", "test/models/23_volatility_model_comparison_test.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

TICKERS, _ = study_universe()
print(f"대상 {len(TICKERS)}종목: {TICKERS}", flush=True)

rows = []
for tk in TICKERS:
    for deper in (False, True):
        t0 = time.time()
        try:
            close = m.load_close(tk)
            r_s = np.log(close).diff().dropna()
            if deper:
                r_s = m.remove_periodicity(r_s)
            r = r_s.to_numpy()
            n = len(r)
            if n < 20000:
                print(f"  {tk} deper={deper} 표본부족(n={n}) 스킵", flush=True)
                continue

            fut, pas = m.targets(r, m.HORIZON)
            valid = np.isfinite(fut) & np.isfinite(pas)
            valid[:7 * m.BPD + 10] = False
            valid_tr = valid & (fut > 0) & (pas > 0)
            idxv = np.where(valid)[0]
            sp_all = int(len(idxv) * m.TRAIN_FRAC)
            split_idx = int(idxv[sp_all])
            tr_mask = valid_tr.copy(); tr_mask[split_idx:] = False
            te_mask = valid.copy(); te_mask[:split_idx] = False
            fit_mask = tr_mask | te_mask
            sp = int(tr_mask.sum())
            y = np.log(fut[fit_mask] + 1e-14)
            rv_act_all = fut[fit_mask]; rv_nai_all = pas[fit_mask]
            ytr, yte = y[:sp], y[sp:]
            ii = np.arange(0, len(yte), m.HORIZON)
            rv_act = rv_act_all[sp:][ii]; rv_nai = rv_nai_all[sp:][ii]
            rv_tr = rv_act_all[:sp]
            f_lo, f_hi = np.percentile(rv_tr, [0.5, 99.5]); f_mean = float(rv_tr.mean())

            gv = m.garch_cond_vol(r, split_idx, dist="t")
            gar_pred = gv[te_mask] * np.sqrt(m.HORIZON)
            p_g, tr_g = m.insanity_filter(gar_pred[ii], f_lo, f_hi, f_mean)
            ev_g = m.evaluate(rv_act, p_g, rv_nai)

            r_pct = r * 100
            theta, ll, info = fit_ms_garch(r_pct, split_idx, n_restarts=2, maxiter=700, max_fit_n=30000)
            ms_vol = np.sqrt(np.clip(info["h_pred"], 0, None)) / 100.0
            ms_pred = ms_vol[te_mask] * np.sqrt(m.HORIZON)
            p_ms, tr_ms = m.insanity_filter(ms_pred[ii], f_lo, f_hi, f_mean)
            ev_ms = m.evaluate(rv_act, p_ms, rv_nai)

            qs = np.quantile(rv_act, [0.2, 0.4, 0.6, 0.8])
            bucket = np.digitize(rv_act, qs)
            q_g = {}; q_ms = {}; q_n = {}
            for q in range(5):
                idx = bucket == q
                q_n[q] = int(idx.sum())
                q_g[q] = m.qlike(rv_act[idx] ** 2, p_g[idx] ** 2) if idx.sum() > 5 else np.nan
                q_ms[q] = m.qlike(rv_act[idx] ** 2, p_ms[idx] ** 2) if idx.sum() > 5 else np.nan

            row = dict(ticker=tk, deper=deper, n=n, split_idx=split_idx,
                       qlike_garch=ev_g["QLIKE"], qlike_ms=ev_ms["QLIKE"],
                       mase_garch=ev_g["MASE"], mase_ms=ev_ms["MASE"],
                       trim_garch=tr_g, trim_ms=tr_ms,
                       p11=info["p11"], p22=info["p22"], nu=info["nu"],
                       omega1=info["omega"][0], omega2=info["omega"][1],
                       alpha1=info["alpha"][0], alpha2=info["alpha"][1],
                       beta1=info["beta"][0], beta2=info["beta"][1])
            for q in range(5):
                row[f"q{q+1}_n"] = q_n[q]
                row[f"q{q+1}_garch"] = q_g[q]
                row[f"q{q+1}_ms"] = q_ms[q]
            rows.append(row)
            dt = time.time() - t0
            print(f"  {tk} deper={deper} 완료 ({dt:.0f}s) QLIKE garch={ev_g['QLIKE']:.3f} "
                  f"ms={ev_ms['QLIKE']:.3f} Q5 garch={q_g[4]:.3f} ms={q_ms[4]:.3f}", flush=True)
        except Exception as e:
            print(f"  {tk} deper={deper} 실패 — {type(e).__name__}: {e}", flush=True)

df = pd.DataFrame(rows)
out = "/tmp/claude-1002/-home-std-jun99120-personal-ai-project-tools-vscode/fc52cb5c-495a-4a72-a905-2a961a95d970/scratchpad/poster/ms_garch_full20_results.csv"
df.to_csv(out, index=False)
print(f"\n저장: {out} ({len(df)}행)")

print("\n===== 요약: Q5(최고변동구간)에서 MS-GARCH가 이긴 비율 =====")
for deper in (False, True):
    sub = df[df["deper"] == deper]
    win = (sub["q5_ms"] < sub["q5_garch"]).sum()
    print(f"주기제거={deper}: {win}/{len(sub)}종목에서 MS-GARCH가 Q5 우세 "
          f"(평균 차이 {float((sub['q5_garch']-sub['q5_ms']).mean()):.3f})")
    win_all = (sub["qlike_ms"] < sub["qlike_garch"]).sum()
    print(f"  전체평균 QLIKE는 {win_all}/{len(sub)}종목에서 MS-GARCH 우세")
