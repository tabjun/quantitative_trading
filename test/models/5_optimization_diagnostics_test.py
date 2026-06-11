# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 5. 최적화 경로 진단 실험
#
# [연구용 스크립트 - Codex 로컬 세션에서 자동 실행 금지]
# 이번 실험은 모델 성능 리더보드를 만들기 위한 것이 아니라,
# 비정상 금융 시계열에서 어떤 objective / head / architecture 조합이
# 쉬운 해(0 수익률, lag-1 복사)에 붕괴하는지를 학습 곡선으로 확인하기 위한 진단용 pass입니다.
#
# 핵심 질문:
# - raw next-close 회귀가 실제로 copy-risk를 키우는가?
# - return target만으로 충분한가, 아니면 Huber / 방향성 penalty / volatility weighting이 필요한가?
# - 같은 objective를 두었을 때 Linear / LSTM / GRU / TCN / Transformer 중 어느 쪽이 더 안정적인가?
#
# 기본 실행 예시:
# - `uv run test/models/5_optimization_diagnostics_test.py --suite objective_probe`
# - `uv run test/models/5_optimization_diagnostics_test.py --suite architecture_probe --epochs 12`
# - `uv run test/models/5_optimization_diagnostics_test.py --suite full_matrix --max-rows 8000 --feature-set text_aware`
#
# 산출물:
# - epoch curve CSV
# - case summary CSV
# - learning curve figure PNG
# - collapse 진단용 Markdown report
#
# 읽는 기준:
# - `collapse_score`가 낮을수록 좋음
# - `variance_ratio`가 0에 가까우면 flat prediction 붕괴 위험
# - `zero_share`가 높으면 0-return shortcut 위험
# - `persistence_gap`이 계속 양수이면 naive copy보다도 못한 상태

# %%
from analysis.optimization_diagnostics import main


if __name__ == "__main__":
    main()
