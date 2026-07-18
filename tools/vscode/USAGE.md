# 원격 VS Code 사용법 (일상 운영 가이드)

`README.md`가 "왜/어떻게 세팅했는지"를 다룬다면, 이 문서는 "평소에 켜고 끄고 접속하는 법"만 간단히 정리한 것.

이 서버(stat5)는 SSH 인바운드(22번)가 방화벽에 막혀 있어 `ssh -L` 터널 방식이 안 된다.
대신 완전히 독립적인 두 가지 방법을 쓴다. **둘은 서로 다른 바이너리, 다른 확장 디렉터리를 쓰는 별개의 서버**라
한쪽에서 확장을 설치해도 다른 쪽에는 안 보인다 (아래 "주의" 참고).

---

## 방법 1: code-server (웹 브라우저, JupyterHub proxy 경유)

브라우저만 있으면 됨. 별도 로컬 설치 불필요.

### 서버에서 상태 확인 / 시작 / 종료

```bash
~/personal_ai_project/tools/vscode/start-vscode.sh status
~/personal_ai_project/tools/vscode/start-vscode.sh start [workdir]   # 기본 workdir: personal_ai_project
~/personal_ai_project/tools/vscode/start-vscode.sh stop
```

- 이미 떠 있으면 `start`는 실패한다 (자동 재시작 안 함) — 먼저 `stop` 해야 새 workdir로 재시작 가능.
- 상태 파일: `~/.local/state/code-server-web/status.env`
- 로그: `~/.local/share/code-server-web/logs/code-server-9999.log`
- 확장/사용자 데이터: `~/.local/share/code-server-web/` (extensions, user-data 분리 저장)

### 접속 (로컬 PC, 브라우저만)

```
https://stat5.kmu.ac.kr:9500/user/std_jun99120/proxy/9999/
```

- JupyterHub 로그인 세션이 있으면 바로 code-server로 리다이렉트됨. 없으면 JupyterHub 로그인 페이지가 먼저 뜸.
- 만약 이 URL이 안 뜨고 계속 로그인 루프에 걸리면: 브라우저에서 `https://stat5.kmu.ac.kr:9500/hub/home` → **Stop My Server → Start My Server**로 JupyterHub singleuser 세션 자체를 재시작 (jupyter_server_proxy 확장이 재시작 시에만 반영되기 때문).

---

## 방법 2: VS Code Remote Tunnels (로컬 VS Code Desktop 직결)

MS 공식 기능. 서버가 아웃바운드로 Microsoft 릴레이에 붙기 때문에 인바운드 22번이 막혀 있어도 동작.

### 서버에서 상태 확인

```bash
~/.local/bin/code tunnel status
```

`"tunnel":"Connected"` 이면 정상. 터널 이름: `stat5-quant`

### 서버에서 시작 / 종료 / 재시작

```bash
# 종료
~/.local/bin/code tunnel kill

# 시작 (백그라운드)
nohup ~/.local/bin/code tunnel --accept-server-license-terms --name stat5-quant \
  > ~/.local/share/code-server-web/logs/vscode-tunnel.log 2>&1 &

# 재시작
~/.local/bin/code tunnel restart
```

- 로그인 토큰은 캐시되므로(`~/.vscode/cli`), 재시작해도 보통 GitHub 재인증 불필요. 로그에 `https://github.com/login/device`와 코드가 다시 뜨면 그때만 재인증.
- 로그: `~/.local/share/code-server-web/logs/vscode-tunnel.log`
- 실제 VS Code 서버 본체(확장/설정 저장 위치): `~/.vscode-server/`

### 문제 해결: "원격 확장 호스트 서버에 연결하지 못했습니다 (WebSocket close 1006)"

로컬에서 연결 시도할 때 이 에러(또는 "원격 환경을 페치할 수 없습니다")가 뜨면, 대부분 **tunnel이 자체 자동 업데이트 도중 멈춘 상태**다. 서버는 계속 살아있으니 로컬/노트북 접속 여부와는 무관하다 (아래 "왜 멈추나" 참고).

진단:
```bash
ls -la ~/.vscode/cli/servers/
```
가장 최근 생성된 `Stable-<hash>/` 안에 `server/` 디렉터리는 있는데 `log.txt`, `pid.txt`가 없다면 그게 원인 — 다운로드/압축해제만 되고 실제 기동은 안 된 상태.

해결 (`code tunnel kill`이 안 먹힐 때가 있으므로 프로세스를 직접 확인):
```bash
pgrep -af "code tunnel"                 # PID 확인
kill <PID들>                             # 안 죽으면 kill -9
pgrep -af "code tunnel"                 # 다 사라졌는지 재확인

~/.local/bin/code tunnel prune          # 기동 안 된 깨진 버전 정리

nohup ~/.local/bin/code tunnel --accept-server-license-terms --name stat5-quant \
  > ~/.local/share/code-server-web/logs/vscode-tunnel.log 2>&1 &
disown

~/.local/bin/code tunnel status         # started_at/last_connected_at이 방금 시각인지 확인
```

**왜 멈추나 (+ 노트북 껐다 켠 것과는 상관없음)**: tunnel은 로컬 노트북이 아니라 서버 자체에서 독립적으로 도는 백그라운드 프로세스라, 클라이언트 접속 여부와 무관하게 서버가 알아서 주기적으로 자체 업데이트를 체크한다. 이번 것도 노트북을 껐다 켠 것과는 무관하고, 서버 쪽에서 자동 업데이트가 진행되다가 중간에 (네트워크 순간 끊김 등으로 추정) 멈춘 것으로 보인다. 재현되면 위 절차 그대로 반복하면 된다.

### 로컬 PC에서 연결

1. VS Code Desktop에 확장 설치: **Remote - Tunnels** (extension id: `ms-vscode.remote-server`)
2. `Ctrl+Shift+P` → `Remote Tunnels: Connect to Tunnel`
3. 로그인 방식 선택 창이 뜨는데 **Microsoft 계정**과 **GitHub 계정** 두 옵션이 나온다 — 반드시 **GitHub**을 선택할 것.
   - 서버에서 `code tunnel` 실행 시 device code 인증을 GitHub로 했기 때문에(로그: `Using GitHub for authentication`), 터널이 GitHub 계정에 연결되어 있음.
   - Microsoft 계정으로 로그인하면 같은 사람이어도 다른 계정 취급이라 터널 목록에 `stat5-quant`가 안 뜬다.
4. 서버에서 device code 인증할 때 쓴 것과 **같은 GitHub 계정**으로 로그인
5. 터널 목록에서 `stat5-quant` 선택 → 연결

### 설치 없이 브라우저만 쓰고 싶으면

```
https://vscode.dev/tunnel/stat5-quant/home/std_jun99120/personal_ai_project
```

---

## 자주 묻는 것

**두 방법을 평소에 둘 다 켜놔도 되나?**
메모리 부담은 거의 없음 (code-server 트리 합계 ~800MB, tunnel ~50MB — 서버 전체 30GB 중 3% 수준). 껐다 켰다 신경 쓸 필요는 없고, 메인으로 쓰는 하나만 정하고 나머지는 편한 대로 두면 됨.

**두 방법이 세션/확장을 공유하나?**
아니다. 완전히 다른 서버 프로세스, 다른 확장 디렉터리다.
- code-server: `~/.local/share/code-server-web/extensions`
- tunnel(공식 vscode-server): `~/.vscode-server/extensions`

한쪽에서 설치한 확장은 다른 쪽에 안 보인다. 같아 보이는 건 두 방법 다 **같은 서버, 같은 프로젝트 폴더**를 열고 있어서 파일/git 상태만 동일하게 보이는 것 (에디터 세션 자체는 독립).
