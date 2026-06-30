#!/bin/bash
set -u

PORT=9999
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUS_FILE="$HOME/.local/state/code-server-web/status.env"
URL="https://stat5.kmu.ac.kr:9500/user/std_jun99120/proxy/${PORT}/"

status=""
pid=""
log_file=""
started_at=""
updated_at=""
data_dir=""
ext_dir=""
workdir=""

if [ -f "$STATUS_FILE" ]; then
    status="$(sed -n 's/^STATUS=//p' "$STATUS_FILE" | tail -n 1)"
    pid="$(sed -n 's/^PID=//p' "$STATUS_FILE" | tail -n 1)"
    log_file="$(sed -n 's/^LOG=//p' "$STATUS_FILE" | tail -n 1)"
    started_at="$(sed -n 's/^STARTED_AT=//p' "$STATUS_FILE" | tail -n 1)"
    updated_at="$(sed -n 's/^UPDATED_AT=//p' "$STATUS_FILE" | tail -n 1)"
    data_dir="$(sed -n 's/^DATA_DIR=//p' "$STATUS_FILE" | tail -n 1)"
    ext_dir="$(sed -n 's/^EXT_DIR=//p' "$STATUS_FILE" | tail -n 1)"
    workdir="$(sed -n 's/^WORKDIR=//p' "$STATUS_FILE" | tail -n 1)"
fi

is_running="no"
if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    is_running="yes"
else
    live_pid="$(pgrep -f "code-server.*127.0.0.1:${PORT}" | head -n 1)"
    if [ -n "${live_pid:-}" ] && kill -0 "$live_pid" 2>/dev/null; then
        pid="$live_pid"
        is_running="yes"
    fi
fi

echo "URL: $URL"
echo "Running: $is_running"
echo "PID: ${pid:-unknown}"
echo "Status file: $STATUS_FILE"
echo "Launcher: $SCRIPT_DIR/start-vscode.sh"

if [ -n "${status:-}" ]; then
    echo "Recorded status: $status"
fi
if [ -n "${started_at:-}" ]; then
    echo "Started at (UTC): $started_at"
fi
if [ -n "${updated_at:-}" ]; then
    echo "Status updated at (UTC): $updated_at"
fi
if [ -n "${workdir:-}" ]; then
    echo "Workdir: $workdir"
fi
if [ -n "${data_dir:-}" ]; then
    echo "User data dir: $data_dir"
fi
if [ -n "${ext_dir:-}" ]; then
    echo "Extensions dir: $ext_dir"
fi
if [ -n "${log_file:-}" ]; then
    echo "Log: $log_file"
fi

if [ "$is_running" = "yes" ]; then
    exit 0
fi

exit 1
