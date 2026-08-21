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
로컬 PC 쪽 확장 설치·로그인 같은 최초 1회 설정은 맨 아래 "로컬 PC 최초 1회 설정" 참고 (이미 해뒀으면 매번 다시 할 필요 없음).

### 매번 쓰는 흐름 (서버 터미널에서)

**1. 터널 시작 (백그라운드)**
```bash
mkdir -p ~/.local/share/code-server-web/logs
nohup ~/.local/bin/code tunnel --accept-server-license-terms --name stat5-quant \
  > ~/.local/share/code-server-web/logs/vscode-tunnel.log 2>&1 &
disown
```
- 로그인 토큰은 캐시되므로(`~/.vscode/cli`), 재시작해도 보통 GitHub 재인증 불필요.
- 로그에 `https://github.com/login/device`와 코드가 새로 뜨면 그때만 재인증 필요.

**2. 실행 확인 (서버 쪽)**
```bash
~/.local/bin/code tunnel status
```
`"tunnel":{"tunnel":"Connected", ...}` 이면 정상. 터널 이름: `stat5-quant`

**3. 로컬 VS Code에서 연결 확인**
왼쪽 사이드바 **원격 탐색기(Remote Explorer)** → `원격(터널/SSH)` → `Tunnels` → `stat5-quant` 옆에 초록 체크 + "연결됨"이 뜨면 정상. 안 보이면 옆의 새로고침(⟳) 클릭.

**4. 작업**
`stat5-quant` 더블클릭(또는 우클릭 → Connect in New Window)해서 원격 폴더 열고 작업.

**5. 다 쓰면 종료 (서버 쪽)**
```bash
~/.local/bin/code tunnel kill
```
이걸 실행하면 로컬 VS Code 연결과 브라우저(vscode.dev) 접속이 둘 다 끊긴다 — 같은 터널 하나를 공유하는 두 접속 경로라 따로 끄는 방법은 없음.

### 상태 확인 / 재시작 (참고용)

```bash
~/.local/bin/code tunnel status    # 상태만 확인
~/.local/bin/code tunnel restart   # 재시작
```

- 로그: `~/.local/share/code-server-web/logs/vscode-tunnel.log`
- 실제 VS Code 서버 본체(확장/설정 저장 위치): `~/.vscode-server/`

### 문제 해결: "원격 확장 호스트 서버에 연결하지 못했습니다 (WebSocket close 1006)"

로컬에서 연결 시도할 때 이 에러(또는 "원격 환경을 페치할 수 없습니다")가 뜨면 원인은 두 단계로 나뉜다.

- **1단계(Transport)**: 로컬 ↔ Microsoft 릴레이 ↔ 서버 `code tunnel` 데몬 사이 통로 연결. 이게 되면 "원격에 연결되었습니다"가 뜬다.
- **2단계(Extension Host)**: 그 통로 위에서 실제 VS Code 서버(확장 실행 담당)를 기동/부착. 여기서 실패하면 1006 에러가 뜬다.

즉 "연결됐다"가 떠도 그 다음 단계에서 막히면 이 에러가 나는 거고, WSL 같은 다른 remote 확장이 끼어든 게 아니다 — Tunnel/SSH/WSL 어떤 remote든 동일한 2단계 구조다.

**(2026-07-18 확인) 원인은 특정 서버 버전(`1.129.1`, commit `8a7abeba`)이 이 서버에서 기동 자체가 안 되는 문제였다.** 로컬 VS Code가 "원격 서버를 업데이트할까요?"라고 물어볼 때 수락하면 이 버전으로 갈아타면서 끊긴다. binary 자체는 정상(`--version` 실행하면 응답함)인데, tunnel 데몬이 실제로 백그라운드 서버로 띄우면 `log.txt`/`pid.txt`가 끝내 안 생기는 채로 멈춘다 (60초 이상 기다려도 마찬가지 — 그냥 느린 게 아니라 진짜 멈추는 것). 두 번 재현 확인함.

**대응: 업데이트 프롬프트가 뜨면 수락하지 말고 "나중에/취소"** — 지금 잘 되는 이전 버전(`1.129.0`, commit `125df4672`)을 계속 쓰면 된다. 나중에 VS Code Desktop이 자체 버전을 올려서 또 이 프롬프트가 뜨면, 그때도 일단 거절하고 이 문서에 새로 적힌 게 없으면 먼저 물어보고 판단할 것.

진단:
```bash
ls -la ~/.vscode/cli/servers/
```
가장 최근 생성된 `Stable-<hash>/` 안에 `server/` 디렉터리는 있는데 `log.txt`, `pid.txt`가 없다 → 그 버전이 기동 실패 중.

이미 업데이트를 수락해서 끊긴 경우 복구 (기존 정상 버전으로 되돌리기):
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
재기동 후 다시 접속했는데 또 업데이트 프롬프트가 뜨면 이번엔 거절할 것 — 그래야 문제되는 버전으로 다시 안 넘어간다.

**왜 멈추나 (+ 노트북 껐다 켠 것과는 상관없음)**: tunnel은 로컬 노트북이 아니라 서버 자체에서 독립적으로 도는 백그라운드 프로세스라, 클라이언트 접속 여부와 무관하게 서버가 알아서 주기적으로 자체 업데이트를 체크한다. 이번 것도 노트북을 껐다 켠 것과는 무관하고, 서버 쪽에서 자동 업데이트가 진행되다가 중간에 (네트워크 순간 끊김 등으로 추정) 멈춘 것으로 보인다. 재현되면 위 절차 그대로 반복하면 된다.

### 로컬 PC 최초 1회 설정 (한 번 해두면 이후엔 "매번 쓰는 흐름"만 반복)

1. VS Code Desktop에 확장 설치: **Remote - Tunnels** (extension id: `ms-vscode.remote-server`)
2. `Ctrl+Shift+P` → `Remote Tunnels: Connect to Tunnel`
3. 로그인 방식 선택 창이 뜨는데 **Microsoft 계정**과 **GitHub 계정** 두 옵션이 나온다 — 반드시 **GitHub**을 선택할 것.
   - 서버에서 `code tunnel` 실행 시 device code 인증을 GitHub로 했기 때문에(로그: `Using GitHub for authentication`), 터널이 GitHub 계정에 연결되어 있음.
   - Microsoft 계정으로 로그인하면 같은 사람이어도 다른 계정 취급이라 터널 목록에 `stat5-quant`가 안 뜬다.
4. 서버에서 device code 인증할 때 쓴 것과 **같은 GitHub 계정**으로 로그인
5. 터널 목록에서 `stat5-quant` 선택 → 연결

로그인은 한 번 하면 로컬에 캐시되므로, 다음부터는 "매번 쓰는 흐름"의 3번(원격 탐색기에서 `stat5-quant` 클릭)만 하면 된다.

### (참고) 브라우저로도 열린다 — 별도 기능 아님

```
https://vscode.dev/tunnel/stat5-quant/home/std_jun99120/personal_ai_project
```

`code tunnel`은 정의상 "vscode.dev에서 접속 가능한 터널"이라, 터널이 켜져 있으면 이 URL로 브라우저 접속도 항상 같이 열려 있다 (GitHub 인증 필요). 로컬 VS Code Desktop 접속과 이 브라우저 접속은 **같은 터널 하나를 공유하는 두 경로**일 뿐, 하나만 켜고 하나만 끄는 옵션은 없다 — 완전히 막고 싶으면 위 5번(`code tunnel kill`)으로 터널 자체를 꺼야 하고, 그러면 둘 다 끊긴다.

---

## 자주 묻는 것

**두 방법을 평소에 둘 다 켜놔도 되나?**
메모리 부담은 거의 없음 (code-server 트리 합계 ~800MB, tunnel ~50MB — 서버 전체 30GB 중 3% 수준). 껐다 켰다 신경 쓸 필요는 없고, 메인으로 쓰는 하나만 정하고 나머지는 편한 대로 두면 됨.

**두 방법이 세션/확장을 공유하나?**
아니다. 완전히 다른 서버 프로세스, 다른 확장 디렉터리다.
- code-server: `~/.local/share/code-server-web/extensions`
- tunnel(공식 vscode-server): `~/.vscode-server/extensions`

한쪽에서 설치한 확장은 다른 쪽에 안 보인다. 같아 보이는 건 두 방법 다 **같은 서버, 같은 프로젝트 폴더**를 열고 있어서 파일/git 상태만 동일하게 보이는 것 (에디터 세션 자체는 독립).
