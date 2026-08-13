#!/usr/bin/env bash
# t8 다중 시드 실행기 — trend_corr의 실행 간 잡음을 정량화한다.
#
# 왜: 2026-08-14 재실행에서 같은 구성·같은 시드·거의 같은 데이터인데도 종목별 trend_corr가
# 최대 +0.0355 움직였다. 10종목 평균이 +0.0167이라 잡음이 신호보다 크다. t8의 판독 기준
# ("여러 종목이 함께 양수면 구조적")이 성립하는지 보려면 먼저 표준오차를 알아야 한다.
#
# 설계: 서로 다른 시드 4개 + 시드 42 반복 1회.
#   - 42 vs 42_repeat  → 시드가 같으므로 차이는 순수 GPU 비결정성
#   - 42/7/123/2026    → 시드 민감도
# t8은 `_base_case`가 seeds[0]만 쓰므로 시드를 순회하지 않는다. 그래서 실행을 반복한다.
#
# 사용법:
#   bash test/scripts/run_t8_multiseed.sh
# 결과: test/results/18b_rerun_verification_20260814/t8_seed<SEED>.csv

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
OUT_DIR="$ROOT/test/results/18b_rerun_verification_20260814"
LEADERBOARD="$ROOT/test/results/15_trend_capture_defense_20260716/t8_multiasset_leaderboard.csv"
LOG_DIR="$ROOT/logs"

# 7월 승계 구성과 일치시킨다 (검증노트 3절 — 기본값으로 돌리면 다른 실험이 된다)
MODEL="ITransformerLike"
OBJECTIVE="huber"
TICKERS="KRW-XRP,KRW-BTC,KRW-DOGE,KRW-ETH,KRW-SOL,KRW-SHIB,KRW-SEI,KRW-XLM,KRW-SUI,KRW-ETC"

# label:seed 쌍. 42를 두 번 돌려 순수 비결정성을 분리한다.
RUNS=("42:42" "42repeat:42" "7:7" "123:123" "2026:2026")

mkdir -p "$OUT_DIR" "$LOG_DIR"

for entry in "${RUNS[@]}"; do
    label="${entry%%:*}"
    seed="${entry##*:}"
    echo "=== [t8-multiseed] label=${label} seed=${seed} 시작 $(date +%H:%M:%S) ==="
    "$PY" test/models/15_trend_capture_defense_test.py \
        --suite t8_multiasset \
        --models "$MODEL" \
        --objective "$OBJECTIVE" \
        --seeds "$seed" \
        --tickers "$TICKERS" \
        > "$LOG_DIR/15_t8_seed${label}.log" 2>&1
    status=$?
    if [ $status -ne 0 ]; then
        echo "[t8-multiseed] label=${label} 실패(exit ${status}) — 로그: $LOG_DIR/15_t8_seed${label}.log"
        continue
    fi
    if [ -f "$LEADERBOARD" ]; then
        cp "$LEADERBOARD" "$OUT_DIR/t8_seed${label}.csv"
        echo "[t8-multiseed] label=${label} 저장: $OUT_DIR/t8_seed${label}.csv"
    else
        echo "[t8-multiseed] label=${label} 리더보드 없음 — 건너뜀"
    fi
done

# 7월 산출물은 이 실행으로 덮어써졌다. 날짜 디렉터리는 그 날짜의 기록이어야 하므로 복원한다.
git -C "$ROOT/.." checkout -- \
    quantitative_trading/test/results/15_trend_capture_defense_20260716/ \
    quantitative_trading/test/images/15_trend_capture_defense_20260716/ 2>/dev/null \
    && echo "[t8-multiseed] 7월 디렉터리 복원 완료"

echo "=== [t8-multiseed] 전체 완료 $(date +%H:%M:%S) ==="
