# 테스트 및 분석 파이프라인 (Test & Analysis Process)

이 문서는 `/test` 폴더 내에서 주피터 노트북(`.ipynb`)을 활용하여 분석을 수행하고, 그 결과를 심층 분석 보고서(MD, PDF)로 자동 변환하여 메일로 전송하는 전체 워크플로우를 정의합니다.

## 📌 1. 워크플로우 개요 (Workflow)
1. **분석 코드 작성:** `/test` 폴더 내의 `.ipynb` 파일에 모델링 및 시각화 코드를 작성합니다.
2. **커널 실행:** 주피터 환경에서 모든 셀을 실행하여 출력 결과(Output)를 노트북 파일에 저장합니다.
3. **고급 보고서 생성 (`enhanced_report_generator.py`):** 
   - 노트북에서 MSE, MAE 지표를 자동 추출하여 요약 테이블을 생성합니다.
   - 현재 가격 대비 오차율(%) 및 알고리즘별 성능 비교 해석을 자동으로 생성합니다.
4. **PDF 변환 (`md_to_pdf.py`):** 
   - 생성된 MD 보고서를 PDF로 변환합니다. 
   - `fpdf2` 라이브러리와 시스템 폰트(맑은 고딕 등)를 사용하여 한국어 및 이미지를 처리합니다.
5. **이메일 자동 발송 (`send_email.py`):** 
   - `.env`에 설정된 정보를 바탕으로 네이버 SMTP를 통해 MD 및 PDF 보고서를 전송합니다.

---

## ✅ 2. 실행 명령어 (Full Pipeline)

모든 분석이 완료된 후, 아래 명령어를 순차적으로 실행하여 보고서를 발행합니다.

```bash
# 1. 고급 마크다운 보고서 생성
python3 test/enhanced_report_generator.py

# 2. PDF 변환 (한국어 폰트 지원)
python3 test/md_to_pdf.py test/analysis_report.md test/analysis_report.pdf

# 3. 이메일 전송 (MD, PDF 첨부)
python3 test/send_email.py
```

---

## 🛠️ 3. 주요 모듈 설명
- `enhanced_report_generator.py`: 노트북의 JSON 데이터를 파싱하여 정량적 지표와 정성적 해석이 포함된 MD 파일을 생성합니다.
- `md_to_pdf.py`: 마크다운의 헤더, 텍스트, 이미지를 해석하여 PDF로 렌더링합니다. (WSL 환경의 Windows 폰트 경로 참조)
- `send_email.py`: 멀티파트 메일을 구성하여 보고서 파일들을 안전하게 전송합니다.

---

## 📝 4. 최근 수행 결과 (2026-05-19)
- **대상**: `1_time_series_test.ipynb` (비트코인 단변량 분석)
- **성능**: ODE-RNN 모델이 가장 낮은 오차율(약 1.6%)을 기록함.
- **자동화**: 분석-보고서-PDF-메일 발송 파이프라인 구축 및 검증 완료.
