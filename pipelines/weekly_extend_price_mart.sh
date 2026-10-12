#!/usr/bin/env bash
# 주간 가격 마트 연장(crontab에서 매주 월요일 03:30 실행).
# 가장 최근 data/upbit_data_YYYYMMDD.db를 원본으로 extend_price_mart.py를 돌려 오늘 날짜 DB를 새로 만든다.
# 기존 DB는 읽기만 하고 지우지 않는다(27번 기준 스냅숏 upbit_data_20261005.db 포함, 각 주 스냅숏은 재현용으로 보존).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
today=$(date +%Y%m%d)
out="data/upbit_data_${today}.db"
if [ -e "$out" ]; then
    echo "[weekly] $(date '+%F %T') 이미 있음: $out"
    exit 0
fi
src=$(ls data/upbit_data_2*.db | sort | tail -1)
tmp="data/.upbit_data_${today}.partial.db"
rm -f "$tmp" "$tmp.wal"
echo "[weekly] $(date '+%F %T') 시작: $src → $out"
.venv/bin/python pipelines/extend_price_mart.py --src "$src" --out "$tmp"
mv "$tmp" "$out"
echo "[weekly] $(date '+%F %T') 완료: $out"
