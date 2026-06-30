#!/bin/bash
set -u

PORT=9999
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_DIR="$HOME/.local/share/code-server-web"
STATE_DIR="$HOME/.local/state/code-server-web"
DATA_DIR="$BASE_DIR/user-data"
EXT_DIR="$BASE_DIR/extensions"
LOG_DIR="$BASE_DIR/logs"
LOG="$LOG_DIR/code-server-${PORT}.log"
STATUS_FILE="$STATE_DIR/status.env"
CS_BIN="$HOME/.local/bin/code-server"
DEFAULT_WORKDIR="$PROJECT_ROOT"
URL="https://stat5.kmu.ac.kr:9500/user/std_jun99120/proxy/${PORT}/"
DEFAULT_EXTENSIONS=(
    "anthropic.claude-code"
    "openai.chatgpt"
    "ms-ceintl.vscode-language-pack-ko"
    "ms-python.python"
    "ms-toolsai.jupyter"
    "charliermarsh.ruff"
    "eamodio.gitlens"
    "usernamehw.errorlens"
)

usage() {
    cat <<EOF
사용법:
  $0 start [workdir]
  $0 stop
  $0 status
EOF
}

timestamp_now() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

write_status() {
    local status="$1"
    local pid="${2:-}"
    local started_at="${3:-}"
    local workdir_value="${4:-$DEFAULT_WORKDIR}"

    mkdir -p "$STATE_DIR"

    cat > "$STATUS_FILE" <<EOF
STATUS=${status}
PID=${pid}
PORT=${PORT}
URL=${URL}
WORKDIR=${workdir_value}
DATA_DIR=${DATA_DIR}
EXT_DIR=${EXT_DIR}
LOG=${LOG}
STARTED_AT=${started_at}
UPDATED_AT=$(timestamp_now)
EOF
}

install_missing_extensions() {
    local extension

    for extension in "${DEFAULT_EXTENSIONS[@]}"; do
        if "$CS_BIN" --user-data-dir "$DATA_DIR" --extensions-dir "$EXT_DIR" --list-extensions 2>/dev/null | grep -Fxq "$extension"; then
            continue
        fi

        echo "확장 설치 중: $extension"
        if ! "$CS_BIN" --user-data-dir "$DATA_DIR" --extensions-dir "$EXT_DIR" --install-extension "$extension" >> "$LOG" 2>&1; then
            echo "확장 설치 실패: $extension"
        fi
    done
}

get_live_pid() {
    pgrep -f "code-server.*127.0.0.1:${PORT}" | head -n 1
}

get_recorded_pid() {
    if [ -f "$STATUS_FILE" ]; then
        sed -n 's/^PID=//p' "$STATUS_FILE" | tail -n 1
    fi
}

print_status_summary() {
    if [ -x "$SCRIPT_DIR/check-vscode.sh" ]; then
        "$SCRIPT_DIR/check-vscode.sh"
        return
    fi

    echo "URL: $URL"
    echo "Status file: $STATUS_FILE"
}

start_server() {
    local requested_workdir="${1:-$DEFAULT_WORKDIR}"
    local existing_pid
    local existing_started_at
    local pid
    local started_at
    local final_status

    mkdir -p "$DATA_DIR" "$EXT_DIR" "$LOG_DIR" "$STATE_DIR"

    if [ ! -x "$CS_BIN" ]; then
        write_status "missing_binary" "" "" "$requested_workdir"
        echo "code-server 실행 파일이 없습니다: $CS_BIN"
        exit 1
    fi

    if [ ! -d "$requested_workdir" ]; then
        write_status "missing_workdir" "" "" "$requested_workdir"
        echo "workdir가 존재하지 않습니다: $requested_workdir"
        exit 1
    fi

    existing_pid="$(get_live_pid)"
    if [ -n "${existing_pid:-}" ] && kill -0 "$existing_pid" 2>/dev/null; then
        existing_started_at=""
        if [ -f "$STATUS_FILE" ]; then
            existing_started_at="$(sed -n 's/^STARTED_AT=//p' "$STATUS_FILE" | tail -n 1)"
        fi
        write_status "running" "$existing_pid" "$existing_started_at" "$requested_workdir"
        echo "에러: 이미 실행 중인 VS Code 웹 세션이 있습니다. (PID=$existing_pid)"
        echo "새 설정을 적용하려면 먼저 '$0 stop' 으로 종료하세요."
        echo "접속 주소: $URL"
        echo "상태 확인: $SCRIPT_DIR/check-vscode.sh"
        exit 1
    fi

    nohup "$CS_BIN" \
      --bind-addr "127.0.0.1:${PORT}" \
      --auth none \
      --disable-update-check \
      --disable-telemetry \
      --user-data-dir "$DATA_DIR" \
      --extensions-dir "$EXT_DIR" \
      "$requested_workdir" \
      > "$LOG" 2>&1 &

    pid="$!"
    started_at="$(timestamp_now)"
    write_status "starting" "$pid" "$started_at" "$requested_workdir"

    echo "시작 중... (PID=$pid)"
    echo "작업 경로: $requested_workdir"

    while kill -0 "$pid" 2>/dev/null; do
        if grep -Eq "HTTP server listening|timed out|error" "$LOG" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if grep -q "HTTP server listening" "$LOG" 2>/dev/null; then
        install_missing_extensions
        write_status "running" "$pid" "$started_at" "$requested_workdir"
        echo "VS Code 실행 완료"
        echo "접속 주소: $URL"
        echo "상태 파일: $STATUS_FILE"
    else
        final_status="failed"
        if ! kill -0 "$pid" 2>/dev/null; then
            final_status="stopped"
        fi
        write_status "$final_status" "$pid" "$started_at" "$requested_workdir"
        echo "시작 실패. 로그 확인: cat $LOG"
        exit 1
    fi
}

stop_server() {
    local pid
    local started_at
    local recorded_workdir

    pid="$(get_recorded_pid)"
    if [ -z "${pid:-}" ] || ! kill -0 "$pid" 2>/dev/null; then
        pid="$(get_live_pid)"
    fi

    recorded_workdir="$DEFAULT_WORKDIR"
    if [ -f "$STATUS_FILE" ]; then
        recorded_workdir="$(sed -n 's/^WORKDIR=//p' "$STATUS_FILE" | tail -n 1)"
    fi

    if [ -z "${pid:-}" ] || ! kill -0 "$pid" 2>/dev/null; then
        write_status "stopped" "" "" "$recorded_workdir"
        echo "실행 중인 VS Code 웹 세션이 없습니다."
        exit 0
    fi

    started_at=""
    if [ -f "$STATUS_FILE" ]; then
        started_at="$(sed -n 's/^STARTED_AT=//p' "$STATUS_FILE" | tail -n 1)"
    fi

    echo "종료 중... (PID=$pid)"
    kill "$pid"

    for _ in 1 2 3 4 5; do
        if ! kill -0 "$pid" 2>/dev/null; then
            write_status "stopped" "" "$started_at" "$recorded_workdir"
            echo "VS Code 웹 세션을 종료했습니다."
            return
        fi
        sleep 1
    done

    write_status "stop_requested" "$pid" "$started_at" "$recorded_workdir"
    echo "종료 신호는 보냈지만 아직 프로세스가 살아 있습니다. 상태를 다시 확인하세요."
    exit 1
}

status_server() {
    print_status_summary
}

command="${1:-}"
case "$command" in
    start)
        start_server "${2:-$DEFAULT_WORKDIR}"
        ;;
    stop)
        stop_server
        ;;
    status)
        status_server
        ;;
    *)
        usage
        exit 1
        ;;
esac

exit 0
