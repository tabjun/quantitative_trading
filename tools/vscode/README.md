# Web VS Code on JupyterHub

이 디렉터리는 KMU JupyterHub 서버에서 `code-server`를 안정적으로 띄우고 관리하기 위한 유틸리티를 담는다.

## 파일

- `start-vscode.sh`: 명시적 `start/stop/status` 인터페이스를 제공하는 실행 스크립트
- `check-vscode.sh`: 현재 웹 VS Code 인스턴스와 상태 파일을 조회하는 상태 확인 스크립트

## 왜 저장소 안으로 옮겼나

- 홈 디렉터리 루트에 스크립트가 있으면 Git 루트가 홈일 때 개인 환경 파일과 섞여 추적 범위가 불명확해진다.
- 이 스크립트는 `quantitative_trading`뿐 아니라 `personal_ai_project` 아래 다른 프로젝트를 열 때도 재사용한다.
- 따라서 저장소 안의 `tools/vscode/`에 두고 문서화하는 편이 재현성과 관리 측면에서 낫다.

## 사용법

기본 시작:

```bash
~/personal_ai_project/tools/vscode/start-vscode.sh start
```

특정 디렉터리로 시작:

```bash
~/personal_ai_project/tools/vscode/start-vscode.sh start ~/personal_ai_project/quantitative_trading
```

상태 확인:

```bash
~/personal_ai_project/tools/vscode/start-vscode.sh status
```

종료:

```bash
~/personal_ai_project/tools/vscode/start-vscode.sh stop
```

## 동작 원칙

- 포트는 `9999`로 고정한다.
- 외부 접속 주소는 `https://stat5.kmu.ac.kr:9500/user/std_jun99120/proxy/9999/` 로 고정한다.
- 이미 세션이 떠 있으면 `start`는 자동 재시작하지 않고 실패한다.
- 상태 파일은 홈 루트가 아니라 `~/.local/state/code-server-web/status.env` 에 저장한다.
- 확장/사용자 데이터는 `~/.local/share/code-server-web/` 아래에 저장해 `/tmp` 초기화 영향을 줄인다.

## 기본 복구 확장

- `anthropic.claude-code`
- `openai.chatgpt`
- `ms-ceintl.vscode-language-pack-ko`
- `ms-python.python`
- `ms-toolsai.jupyter`
- `charliermarsh.ruff`
- `eamodio.gitlens`
- `usernamehw.errorlens`

## 참고

- `quantitative_trading/.vscode/settings.json`
- `quantitative_trading/.vscode/extensions.json`

프로젝트별 VS Code 기본값은 위 두 파일이 담당하고, 이 디렉터리의 스크립트는 웹 VS Code 서버 수명주기를 담당한다.
