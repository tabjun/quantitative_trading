#!/bin/bash
# VS Code Remote Tunnel (방법 2) 제어 스크립트.
# 사용법: ./remote_tunnel.sh {start|stop|restart|status}
set -u

BIN="$HOME/.local/bin/code"
NAME="stat5-quant"
PATTERN="code tunnel --accept-server-license-terms --name ${NAME}"
LOG_DIR="$HOME/.local/share/code-server-web/logs"
LOG="$LOG_DIR/vscode-tunnel.log"
CONNECT_TIMEOUT=20   # start 후 Connected 될 때까지 최대 대기(초)
KILL_TIMEOUT=10       # stop 시 프로세스 종료 확인 최대 대기(초)

usage() {
    cat <<EOF
사용법:
  $0 start    # 터널 시작 (이미 떠 있으면 에러)
  $0 stop     # 터널 종료 (로컬 데스크톱/브라우저 접속 둘 다 끊김)
  $0 restart  # stop 후 start
  $0 status   # 현재 상태 확인
EOF
}

get_pids() {
    pgrep -f "$PATTERN" 2>/dev/null
}

# code tunnel status는 프로세스가 죽어있거나 락이 꼬이면 응답 없이 멈출 수 있어
# timeout으로 반드시 끊어준다.
raw_status() {
    timeout 8 "$BIN" tunnel status 2>&1
}

print_status() {
    local pids
    pids="$(get_pids)"

    if [ -z "$pids" ]; then
        echo "[중지됨] 실행 중인 tunnel 프로세스가 없습니다."
        return 1
    fi

    local pid_count
    pid_count="$(echo "$pids" | wc -l)"
    if [ "$pid_count" -gt 1 ]; then
        echo "[경고] tunnel 프로세스가 ${pid_count}개 중복 실행 중입니다 (PID: $(echo "$pids" | tr '\n' ' '))."
        echo "        'code tunnel status'가 응답 없이 멈추는 원인이 될 수 있습니다. 'restart'로 정리하세요."
        return 2
    fi

    echo "프로세스: PID=$pids (실행 중)"

    local out
    out="$(raw_status)"
    if [ -z "$out" ]; then
        echo "[경고] 'code tunnel status'가 ${CONNECT_TIMEOUT}초 내 응답하지 않았습니다 (락 충돌 의심). 'restart'로 정리하세요."
        return 2
    fi

    if echo "$out" | grep -q '"tunnel":"Connected"'; then
        echo "[연결됨] $out"
        return 0
    else
        echo "[미연결] $out"
        echo "로그 확인: tail -n 40 $LOG"
        return 3
    fi
}

do_start() {
    local existing
    existing="$(get_pids)"
    if [ -n "$existing" ]; then
        echo "에러: 이미 실행 중인 tunnel 프로세스가 있습니다 (PID: $(echo "$existing" | tr '\n' ' '))."
        echo "새로 시작하려면 먼저 '$0 stop' 또는 '$0 restart'를 쓰세요."
        exit 1
    fi

    mkdir -p "$LOG_DIR"

    nohup "$BIN" tunnel --accept-server-license-terms --name "$NAME" \
        > "$LOG" 2>&1 &
    disown

    echo "시작 중... (PID=$!)"

    local waited=0
    while [ "$waited" -lt "$CONNECT_TIMEOUT" ]; do
        if echo "$(raw_status)" | grep -q '"tunnel":"Connected"'; then
            echo "성공: 터널이 연결되었습니다."
            print_status
            echo ""
            echo "다음: 로컬 VS Code → 원격 탐색기 → Tunnels → ${NAME} 에서 연결 확인"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    echo "실패: ${CONNECT_TIMEOUT}초 내에 연결되지 않았습니다."
    echo "로그 확인: tail -n 40 $LOG"
    exit 1
}

do_stop() {
    local pids
    pids="$(get_pids)"
    if [ -z "$pids" ]; then
        echo "이미 중지된 상태입니다 (실행 중인 프로세스 없음)."
        return 0
    fi

    echo "종료 중... (PID: $(echo "$pids" | tr '\n' ' '))"
    "$BIN" tunnel kill >/dev/null 2>&1

    local waited=0
    while [ "$waited" -lt "$KILL_TIMEOUT" ]; do
        if [ -z "$(get_pids)" ]; then
            echo "성공: 터널을 종료했습니다."
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    echo "'code tunnel kill'로 안 죽어서 강제 종료합니다."
    # shellcheck disable=SC2046
    kill -9 $(get_pids) 2>/dev/null
    sleep 1

    if [ -z "$(get_pids)" ]; then
        echo "성공: 강제 종료했습니다."
    else
        echo "실패: 프로세스가 여전히 살아 있습니다 (PID: $(get_pids | tr '\n' ' '))."
        exit 1
    fi
}

do_restart() {
    do_stop
    sleep 1
    do_start
}

command="${1:-}"
case "$command" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_restart
        ;;
    status)
        print_status
        ;;
    *)
        usage
        exit 1
        ;;
esac
