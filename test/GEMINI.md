# Test Environment Instructions (Gemini AI 자동화 규칙)

이 파일은 Gemini CLI 에이전트가 `test/` 디렉토리 내에서 작업할 때 **항상(세션 초기화 이후에도) 자동으로 읽고 적용하는 강제 규칙**입니다.

## 1. 이중 코드 관리 체계 자동화 (Dual-File Strategy)
사용자가 명시적으로 요청하지 않더라도, 에이전트가 `test/models/` 내의 `.ipynb` 파일을 수정하거나 분석 코드를 작성한 경우 **반드시 다음 작업을 자동으로 수행**해야 합니다.
1. `ipynb-py-convert <수정된파일.ipynb> <수정된파일.py>` 명령어를 실행하여 `.py` 파일을 동기화합니다.
2. 생성된 `.py` 파일 최상단에 아래의 주석 헤더가 존재하는지 확인하고, 없다면 추가합니다.
```python
# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual execution should be performed in the Jupyter Notebook (.ipynb).
```

## 3. 의존성 패키지 안내 자동화
에이전트가 `test/` 내 코드에서 기존에 사용되지 않던 새로운 라이브러리(`import` 문 추가 등)를 도입한 경우, **반드시 다음 작업을 수행**해야 합니다.
1. `test/README.md`의 **[의존성 패키지]** 섹션에 해당 라이브러리를 업데이트하여 관리합니다.
2. 현재 응답의 마지막 부분에 사용자가 터미널(또는 주피터 셀)에서 즉시 실행할 수 있는 **`pip install` 명령어를 명시적으로 안내**합니다.
3. 생성된 `.py` 및 `.ipynb` 파일 최상단 주석에도 해당 설치 명령어를 포함합니다.
