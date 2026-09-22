"""TAR-GARCH를 top20 전체·2조건으로 실행. GARCH-t와 비교."""
import warnings; warnings.filterwarnings("ignore")
import sys, time, importlib.util
import numpy as np
import pandas as pd

SP = "/tmp/claude-1002/-home-std-jun99120-personal-ai-project-tools-vscode/fc52cb5c-495a-4a72-a905-2a961a95d970/scratchpad/poster"
sys.path.insert(0, SP)
from tar_garch import fit_tar_garch

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

            n_ = len(r)
            switch_lagged = np.empty(n_)
            switch_lagged[0] = np.nan_to_num(pas[0], nan=0.0)
            switch_lagged[1:] = np.nan_to_num(pas[:-1], nan=0.0)
            r_pct = r * 100
            tau_candidates = np.quantile(switch_lagged[tr_mask], [0.6, 0.75, 0.9])

            theta, tau, ll, info = fit_tar_garch(r_pct, switch_lagged, split_idx, tau_candidates,
                                                  n_restarts=1, maxiter=400)
            tar_vol = np.sqrt(np.clip(info["h_pred"], 0, None)) / 100.0
            tar_pred = tar_vol[te_mask] * np.sqrt(m.HORIZON)
            p_tar, tr_tar = m.insanity_filter(tar_pred[ii], f_lo, f_hi, f_mean)
            ev_tar = m.evaluate(rv_act, p_tar, rv_nai)

            qs = np.quantile(rv_act, [0.2, 0.4, 0.6, 0.8])
            bucket = np.digitize(rv_act, qs)
            q_g = {}; q_tar = {}; q_n = {}
            for q in range(5):
                idx = bucket == q
                q_n[q] = int(idx.sum())
                q_g[q] = m.qlike(rv_act[idx] ** 2, p_g[idx] ** 2) if idx.sum() > 5 else np.nan
                q_tar[q] = m.qlike(rv_act[idx] ** 2, p_tar[idx] ** 2) if idx.sum() > 5 else np.nan

            row = dict(ticker=tk, deper=deper, n=n, split_idx=split_idx,
                       qlike_garch=ev_g["QLIKE"], qlike_tar=ev_tar["QLIKE"],
                       mase_garch=ev_g["MASE"], mase_tar=ev_tar["MASE"],
                       trim_garch=tr_g, trim_tar=tr_tar, tau=tau, nu=info["nu"],
                       omega1=info["omega"][0], omega2=info["omega"][1],
                       alpha1=info["alpha"][0], alpha2=info["alpha"][1],
                       beta1=info["beta"][0], beta2=info["beta"][1])
            for q in range(5):
                row[f"q{q+1}_n"] = q_n[q]
                row[f"q{q+1}_garch"] = q_g[q]
                row[f"q{q+1}_tar"] = q_tar[q]
            rows.append(row)
            dt = time.time() - t0
            print(f"  {tk} deper={deper} 완료 ({dt:.0f}s) QLIKE garch={ev_g['QLIKE']:.3f} "
                  f"tar={ev_tar['QLIKE']:.3f} Q5 garch={q_g[4]:.3f} tar={q_tar[4]:.3f}", flush=True)
        except Exception as e:
            print(f"  {tk} deper={deper} 실패 — {type(e).__name__}: {e}", flush=True)

df = pd.DataFrame(rows)
out = f"{SP}/tar_garch_full20_results.csv"
df.to_csv(out, index=False)
print(f"\n저장: {out} ({len(df)}행)")

print("\n===== 요약: Q5(최고변동구간)에서 TAR-GARCH가 이긴 비율 =====")
for deper in (False, True):
    sub = df[df["deper"] == deper]
    win = (sub["q5_tar"] < sub["q5_garch"]).sum()
    print(f"주기제거={deper}: {win}/{len(sub)}종목에서 TAR-GARCH가 Q5 우세 "
          f"(평균 차이 {float((sub['q5_garch']-sub['q5_tar']).mean()):.3f})")
    win_all = (sub["qlike_tar"] < sub["qlike_garch"]).sum()
    print(f"  전체평균 QLIKE는 {win_all}/{len(sub)}종목에서 TAR-GARCH 우세")
