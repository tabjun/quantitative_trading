# 연구 실험 공간 안내

`test/`는 운영 프레임워크가 아니라 연구 실험, 노트북, 결과물, 메모를 두는 공간입니다.

## 1. 폴더 역할

- `models/`: 연구용 `.ipynb` 원본과 동명 `.py` 미러
- `results/`: Markdown 보고서, 요약표, 선택적 CSV
- `research_materials/`: 문헌 메모, 실험 설계, 교수님 공유 브리프
- `images/`: 노트북 출력 그림과 보고서 삽입용 이미지
- `scripts/`: 재사용 가능한 보조 유틸리티만 보관

## 2. 공통 규칙

- 새 실험은 `test/models/*.ipynb`부터 만든다.
- 같은 이름의 `.py` 미러를 함께 유지한다.
- 후속 연구가 기존 번호 실험의 의미를 바꾸면, 기존 파일을 덮어쓰지 않고 새 번호 실험(`5 -> 6 -> 7`)으로 분리한다.
- 무거운 연구 실행은 학교 서버 커널이나 승인된 원격 환경에서 한다.
- 노트북 결과는 기본적으로 `plt.show()`와 셀 출력으로 본다. `savefig()`나 서버 저장 CSV/Markdown은 예외적으로만 쓴다.
- 루트 프레임워크로 올릴 로직은 `analysis/`, `contexts/`, `marts/`, `pipelines/`로 옮긴다.
- 노트북을 파일별로 하나씩 실행한 뒤에는 서버의 `Kernel -> Shut Down All Kernels`로 이전 커널을 완전히 종료하고 다음 파일을 연다.
- 원격 Jupyter 서버에 다시 붙을 때는 서버 `IP:port`와 `token`으로 접속하며, 커널 상태가 꼬이면 재연결보다 기존 커널 종료를 먼저 한다.

## 3. 현재 주요 실험

- `1_time_series_test.ipynb`: 기초 시계열 모델 비교
- `2_time_series_advance_test.ipynb`: 고급 시계열 아키텍처 비교
- `3_time_series_multi_ticker_test.ipynb`: KRW 다종목 실험
- `4_text_independent_variable_analysis.ipynb`: 텍스트 독립변수 결합 실험
- `5_optimization_diagnostics_test.ipynb`: 최적화 경로와 shortcut collapse 진단
- `6_optimization_stabilization_test.ipynb`: target, normalization, loss, model selection 안정화 실험
- `7_optimization_breadth_expansion_test.ipynb`: 6번 이후 확장 실험의 자원 인식형 stage plan
- `8_optimization_breadth_training_test.ipynb`: 7번 계획을 실제 GPU 학습/시각화/붕괴 진단으로 연결한 breadth training 실험. 알고리즘뿐 아니라 preprocessing, normalization, loss, optimizer/scheduler, gradient policy, ensemble 축을 함께 비교한다.
- `9_preprocessing_uncertainty_diagnostics_test.ipynb`: 8번의 전 모델 persistence 미달 결과를 바탕으로 전처리 조합, seed ensemble, conformal interval, 모델 용량과 Double Descent 가능성을 진단한다.

## 4. 4번부터 8번까지의 연구 흐름

- `4번`은 텍스트와 외생변수가 가격 예측에 실제 정보 가치를 주는지 보는 실험이다.
- `5번`은 objective, target, architecture가 직전가 복사나 0 수익률 같은 쉬운 해로 붕괴하는지 보는 진단 실험이다.
- `6번`은 독립변수와 데이터마트를 본격적으로 붙이기 전에 target, normalization, loss, model selection 기준을 안정화하는 실험이다.
- `7번`은 실제 학습 결과가 아니라, 6번 안정화 이후 확장 실험을 어떤 자원 profile과 stage로 실행할지 정리한 계획/점검 산출물이다.
- `8번`은 7번에서 빠진 실제 학습 backend를 새 번호로 분리한 실험이며, 모델군 확장만이 아니라 전처리/정규화/손실함수/최적화/기울기 안정화/앙상블 조합까지 서버 GPU에서 비교한다.
- `9번`은 8번에서 관찰된 0수익률 평탄화와 출력 분산 폭주를 분리하기 위해 극단값·heavy-tail·추세·주파수·변동성 전처리를 조합하고, 예측구간과 모델 용량 변화까지 확인한다.
- 문헌 기반 논문화 방향과 후속 알고리즘 후보는 `test/research_materials/forecasting_methodology_literature_review_20260613.md`를 본다.
- 세부 해석은 각 노트북의 결과 셀과 `test/results/*.md` 보고서를 우선 본다.
- 설계 메모와 참고문헌은 `test/experiment_specs/`에 둔다.

## 5. 실행과 산출물

- 기본 실행 예시는 각 연구 노트북 안의 셀과 `test/models/*.py`에 둔다.
- 결과 해석의 1차 원본은 `ipynb` 출력 셀이다.
- `8_optimization_breadth_training_test`는 기본적으로 PNG/CSV/Markdown 파일을 서버에 저장하지 않고, 노트북 inline 출력만 남긴다. `savefig()`는 기본 경로가 아니라 예외 경로로만 사용한다. 보고서용 이미지 추출은 `test/scripts/extract_notebook_images.py` 같은 보조 도구로 처리한다.
- `test/scripts/extract_notebook_images.py`는 노트북 출력 그림 추출용 보조 유틸리티다.
- `results/`는 보고서와 필요 시 CSV를 둔다.
- 새 보고서는 이전 보고서를 참고하더라도 알고리즘/손실함수/정규화/지표/그래프 읽는 법을 다시 적는 독립 문서로 작성한다.
- `test/README.md`에는 폴더 안내만 유지하고, 긴 연구 서술은 노트북/보고서로 보낸다.

## 6. 서버 환경 구성

이 저장소는 반복되는 서버 환경 구축을 스크립트로 고정한다.

- `test/scripts/bootstrap_uv_313.sh`: `uv`로 Python 3.13 환경을 새로 만들고, `uv sync`, CUDA 체크, PyTorch wheel 설치, Jupyter kernel 등록까지 한 번에 수행한다.
- `test/scripts/bootstrap_uv_312.sh`: `uv`로 Python 3.12 환경을 새로 만들고 같은 순서로 정리한다.
- `test/scripts/bootstrap_venv_313.sh`: 서버에 `python3.13`이 실제로 있을 때만 plain `venv` 방식으로 같은 작업을 수행한다.
- `test/scripts/bootstrap_venv_312.sh`: 서버에 `python3.12`가 실제로 있을 때만 plain `venv` 방식으로 같은 작업을 수행한다.
- 모든 스크립트는 고정 `.venv`를 덮어쓰지 않고 `.venvs/<env_name>` 형태로 새 환경을 만든다.
- 기본 환경 이름은 날짜/시간이 붙은 형태로 자동 생성되며, 실행 전에 기존 env 디렉터리와 Jupyter kernel 목록도 같이 출력한다.
- 각 스크립트는 설치 후 최소 smoke test를 자동 수행한다. 기본 import(`numpy`, `pandas`, `duckdb`, `statsmodels`, `optuna`, `torch`, `ipykernel`, `openai`, `google.generativeai`, `fastdtw`)와 `torch.cuda.is_available()` 확인까지 포함한다.

### 학교 서버 JupyterLab 저버전 호환 고정값

학교 서버에서는 커널 Python이나 CUDA가 정상이더라도, 상위 JupyterLab/JupyterHub 버전이 낮으면 최신 widget/kernel stack과 충돌할 수 있다. 이 저장소에서는 다음 조합을 **known-good compatibility set**으로 취급한다.

```bash
UV_NO_CONFIG=1 uv pip install --reinstall \
  "ipykernel==6.29.5" \
  "jupyter_client==8.6.3" \
  "traitlets==5.14.3" \
  "pyzmq==26.2.1" \
  "ipywidgets==8.1.8" \
  "jupyterlab_widgets"
```

- 이 조합은 "커널 Python은 정상인데 JupyterLab 쪽 버전이 낮아서 커널 시작/위젯 호환이 꼬이는 경우"를 기준으로 고정한 값이다.
- bootstrap 스크립트는 위 버전 세트를 기본으로 재설치한다.
- 서버에서 이미 env가 살아 있고 `ipywidgets`/`jupyterlab_widgets` 계열만 부족한 것이 확인되면, env 전체를 다시 만들기보다 위 명령만 추가 수행해도 된다.
- 반대로 `torch`, `cuda`, `ipykernel`까지 모두 깨졌다면 새 env 재구축을 우선한다.

실행 예시는 다음처럼 쓴다.

```bash
bash test/scripts/bootstrap_uv_312.sh
bash test/scripts/bootstrap_uv_313.sh
bash test/scripts/bootstrap_venv_312.sh
bash test/scripts/bootstrap_venv_313.sh
```

```bash
# 이름을 직접 주고 싶을 때
ENV_NAME=my_quant_313 KERNEL_NAME=my_quant_313 bash test/scripts/bootstrap_uv_313.sh
```

- 학교 서버에서는 먼저 `git pull --rebase origin <branch>`로 최신 브랜치를 맞추고, 그다음 환경 스크립트를 실행한다.
- 환경이나 노트북을 수정한 뒤에는 `git add`, `git commit`, `git push origin <branch>` 순서로 남긴다.
- 커널을 파일별로 바꿔 가며 실행할 때는 항상 `Kernel -> Shut Down All Kernels`를 먼저 수행한다.
- JupyterLab에서 커널이 안 보이면, 스크립트가 만든 `Python 3.12 (Quant Stat) [env_name]` 또는 `Python 3.13 (Quant Stat) [env_name]` 커널을 다시 선택한다.
- 스크립트 실행이 끝나면 해당 env 경로, 커널 이름, 제거 명령(`rm -rf ...`, `jupyter kernelspec uninstall ...`)도 같이 출력된다.
- smoke test가 실패하면 그 환경은 바로 쓰지 말고, 실패한 import 이름이나 CUDA 체크 메시지를 기준으로 다시 점검한다.

### 로컬 VSCode에서 학교 Jupyter 서버 커널 연결

로컬 VSCode에서는 서버의 `.venv` 경로를 직접 넣지 않고, 실행 중인 Jupyter 서버 URL로 접속한 뒤 등록된 Jupyter 커널을 선택한다.

```bash
# 1. 학교 JupyterLab 터미널에서 실행 중인 서버 URL 확인
jupyter server list
```

출력 예시는 다음과 같다.

```text
Currently running servers:
http://127.0.0.1:56205/user/std_jun99120/?token=<TOKEN> :: /home/std_jun99120
```

VSCode에는 `127.0.0.1` 주소를 그대로 넣지 않고, 학교 JupyterHub 외부 주소로 바꿔 입력한다.

```text
https://stat5.kmu.ac.kr:9500/user/std_jun99120/?token=<TOKEN>
```

그다음 VSCode에서 다음 순서로 연결한다.

```text
Select Kernel
-> Existing Jupyter Server
-> 위 서버 URL 입력
-> 등록된 커널 선택
```

서버에 등록된 커널 목록은 다음 명령어로 확인한다.

```bash
jupyter kernelspec list
```

특정 커널이 실제 어떤 Python 환경을 사용하는지 확인하려면 다음처럼 `kernel.json`을 조회한다.

```bash
cat ~/.local/share/jupyter/kernels/quant313/kernel.json
```

정상이라면 `argv`에 원하는 uv/venv Python 경로가 들어 있어야 한다.

```text
/home/std_jun99120/personal_ai_project/quantitative_trading/.venv/bin/python
```

즉, 연결 구조는 다음과 같다.

```text
로컬 VSCode
-> 학교 Jupyter 서버 URL
-> 서버에 등록된 Jupyter 커널
-> 서버 내부 uv/venv Python
```

원격 연결 문제를 볼 때는 다음도 함께 확인한다.

- VSCode 로그에 `Password ... was invalid`가 반복되면, 저장된 비밀번호/세션이 꼬였다는 뜻일 수 있다. `Existing Jupyter Server`를 다시 선택하고 `jupyter server list`에서 얻은 **새 token URL**로 재연결한다.
- 노트북 메타데이터에 오래된 커널 이름이 박혀 있으면, 환경을 지운 뒤에도 예전 커널을 다시 잡으려 할 수 있다. 이 경우 VSCode에서 다른 커널을 직접 다시 선택한다.
- 커널이 `Started session` 뒤 `Canceled future for execute_request message before replies were done`로 멈추면, 서버에서 먼저 `jupyter kernelspec list`, 해당 `kernel.json`, 그리고 `.../bin/python -c "import ipykernel, torch; print(torch.cuda.is_available())"`를 확인한다.
- `ipykernel`, `torch`, `cuda`는 정상이지만 widget 계열만 깨져 있고 학교 서버 JupyterLab 버전이 낮다면, 위의 compatibility set으로 다시 맞춘 뒤 커널을 재선택한다.

### 장시간 환경 스크립트 백그라운드 실행

JupyterLab 터미널에서 오래 걸리는 환경 구성 스크립트는 `nohup`으로 백그라운드 실행한다. 이렇게 하면 로컬 브라우저나 VSCode 연결이 끊겨도 서버에서 작업이 계속 진행된다.

#### 단일 스크립트 실행

```bash
cd ~/personal_ai_project/quantitative_trading

# logs 디렉터리가 없으면 생성한다. 이미 있으면 그대로 둔다.
mkdir -p logs

LOG="logs/bootstrap_uv_313_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S).log"

nohup bash test/scripts/bootstrap_uv_313.sh > "$LOG" 2>&1 &

echo $! | tee logs/bootstrap_uv_313.pid
echo "$LOG" | tee logs/bootstrap_uv_313.latest

tail -f "$LOG"
```

로그 확인을 멈출 때는 `Ctrl + C`를 누른다. 이는 `tail -f`만 종료하며, `nohup`으로 실행된 스크립트는 계속 실행된다.

```bash
ps -p "$(cat logs/bootstrap_uv_313.pid)" -o pid,stat,etime,%cpu,%mem,cmd
disown %1
```

#### 3.13 완료 후 3.12 순차 실행

두 환경을 동시에 만들면 다운로드, 디스크, 캐시 작업이 겹쳐 느려질 수 있으므로 순차 실행을 기본으로 한다. 아래 명령은 이미 실행 중인 3.13 작업이 끝난 뒤 3.12 작업을 자동으로 시작한다.

```bash
cd ~/personal_ai_project/quantitative_trading

# 위 단일 실행 블록에서 이미 mkdir -p logs를 실행했다면 생략해도 된다.
mkdir -p logs

LOG="logs/bootstrap_uv_312_after_313_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S).log"

nohup bash -lc '
cd ~/personal_ai_project/quantitative_trading
while kill -0 "$(cat logs/bootstrap_uv_313.pid)" 2>/dev/null; do
  sleep 60
done
bash test/scripts/bootstrap_uv_312.sh
' > "$LOG" 2>&1 &

echo $! | tee logs/bootstrap_uv_312.pid
echo "$LOG" | tee logs/bootstrap_uv_312.latest
```

#### 실행 상태 확인

```bash
cd ~/personal_ai_project/quantitative_trading

ps -p "$(cat logs/bootstrap_uv_313.pid)" -o pid,stat,etime,%cpu,%mem,cmd
ps -p "$(cat logs/bootstrap_uv_312.pid)" -o pid,stat,etime,%cpu,%mem,cmd

tail -n 80 "$(cat logs/bootstrap_uv_313.latest)"
tail -n 80 "$(cat logs/bootstrap_uv_312.latest)"
```

#### 완료 확인

```bash
jupyter kernelspec list
find . -maxdepth 4 -name "pyvenv.cfg" -print | sort
```

#### 실행 중단

```bash
kill "$(cat logs/bootstrap_uv_313.pid)"
kill "$(cat logs/bootstrap_uv_312.pid)"
```

`nohup: ignoring input`은 정상 메시지다. 백그라운드 작업이 키보드 입력을 받지 않고 실행된다는 의미다.

## 7. 참고 문서

- [루트 README](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/README.md)
- [AGENTS.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/AGENTS.md)
- [process.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/process.md)
- [history.md](/c:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/history.md)
