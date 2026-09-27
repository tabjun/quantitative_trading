"""레짐전환 2종(MS-GARCH, TAR-GARCH)을 기존 16개 모델 20종목 평균 QLIKE 순위에 끼워본다.
계산식이 model_comparison.csv와 동일함(재계산 GARCH-t 값이 원본과 일치)을 확인 후 병합."""
import pandas as pd

ROOT = "test/results/23_model_comparison_20260908"
ms = pd.read_csv(f"{ROOT}/regime_switching_prelim/ms_garch_full20_results.csv")
tar = pd.read_csv(f"{ROOT}/regime_switching_prelim/tar_garch_full20_results.csv")
main = pd.read_csv(f"{ROOT}/model_comparison.csv")

piv = main.groupby(["모델", "주기제거"])["QLIKE"].mean().unstack()

rows = []
for name, df, col in [("MS-GARCH", ms, "qlike_ms"), ("TAR-GARCH", tar, "qlike_tar")]:
    for deper in (False, True):
        sub = df[df["deper"] == deper]
        g_check = sub["qlike_garch"].mean()
        orig = piv.loc["GARCH-t", deper]
        assert abs(g_check - orig) < 0.01, f"GARCH-t 재계산 불일치: {g_check} vs {orig}"
        rows.append({"모델": name, "주기제거": deper, "QLIKE": sub[col].mean()})

merged = pd.concat([piv.reset_index().melt(id_vars="모델", var_name="주기제거", value_name="QLIKE"),
                     pd.DataFrame(rows)], ignore_index=True)

for deper in (False, True):
    sub = merged[merged["주기제거"] == deper].sort_values("QLIKE")
    print(f"\n===== 주기제거={deper} — 18개 모델(16+레짐전환 2종) 순위 =====")
    for i, r in enumerate(sub.itertuples(), 1):
        print(f"{i:2d}. {r.모델:<18s} {r.QLIKE:.3f}")

merged.to_csv(f"{ROOT}/regime_switching_prelim/rank_comparison_18models.csv", index=False)
print(f"\n저장: {ROOT}/regime_switching_prelim/rank_comparison_18models.csv")
