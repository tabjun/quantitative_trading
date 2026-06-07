import os
import smtplib
from email.mime.text import MIMEText


DEFAULT_REPORT_FILENAME = "AI_trading_youtube_upbit_report_20260607.md"
DEFAULT_REPORT_URL = (
    "https://github.com/tabjun/personal_ai_project/blob/stock/"
    f"quantitative_trading/AI_trading_temporary/{DEFAULT_REPORT_FILENAME}"
)
DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=5avgkEHjBeY"


def load_env_file(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def normalize_naver_sender(sender):
    if sender and "@" not in sender:
        return f"{sender}@naver.com"
    return sender


def build_body(report_filename, report_url, youtube_url):
    return f"""손낙훈 교수님께

안녕하세요. 윤태준입니다.

요청하신 유튜브 영상의 핵심 관점(검증 가능한 규칙, 거래 비용 반영, 손절 및 리스크 통제)을 참고하여, 업비트 15분봉 데이터 기반 AI 트레이딩 모의투자 결과를 다시 정리해 전달드립니다.

결과 보고서는 다음 링크에서 확인할 수 있습니다.
[{report_filename}]({report_url})

내용: 투자에 대한 내용 정리

참고 규칙:
- 초기 자본: 10,000,000원
- 대상 데이터: Upbit KRW-BTC 15분봉, 2023-05-21 03:45 ~ 2026-05-20 03:30
- 진입 규칙: SMA5 상향 교차 또는 20봉 고점 돌파와 거래량 확인
- 청산 규칙: -2% 손절 또는 SMA5 하향 교차
- 거래 비용: 업비트 수수료 0.05%와 슬리피지 0.02%를 매수/매도 각각 반영
- 포지션 규모: 1회 진입 3,000,000원

참고 유튜브 링크:
{youtube_url}

핵심 결과:
- 최종 평가 자산: 2,954,964원
- 누적 손익: -7,045,036원
- 누적 수익률: -70.45%
- 최대낙폭(MDD): -70.45%
- 거래 횟수: 3,704회
- 청산 거래 기준 승률: 21.60%
- 동일 기간 단순 보유 수익률: +214.69%

해석 요약:
이번 테스트는 영상의 취지처럼 "AI 신호를 무조건 신뢰하는 것"이 아니라, 실제 거래 비용과 손절 규칙을 포함했을 때 규칙 기반 AI 트레이딩이 얼마나 견디는지를 확인하는 용도입니다. 결과적으로 잦은 단기 진입과 청산이 수수료 및 슬리피지에 취약했고, 단순 보유 대비 성과가 크게 낮았습니다. 따라서 실전 적용 전에는 매매 빈도 제한, 변동성 필터, 추세 강도 필터, 거래비용 대비 기대수익 기준을 추가하는 것이 필요합니다.

감사합니다.
윤태준 드림
"""


def send_professor_report():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    load_env_file(os.path.join(root, "test", ".env"))
    load_env_file(os.path.join(os.path.dirname(__file__), ".env"))

    sender_email = normalize_naver_sender(os.environ.get("NAVER_EMAIL_ID"))
    sender_password = os.environ.get("NAVER_APP_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    if not sender_email or not sender_password or not receiver_email:
        raise RuntimeError(
            "Missing NAVER_EMAIL_ID, NAVER_APP_PASSWORD, or RECEIVER_EMAIL."
        )

    report_filename = os.environ.get("REPORT_FILENAME", DEFAULT_REPORT_FILENAME)
    report_url = os.environ.get("REPORT_URL", DEFAULT_REPORT_URL)
    youtube_url = os.environ.get("YOUTUBE_URL", DEFAULT_YOUTUBE_URL)

    msg = MIMEText(
        build_body(report_filename, report_url, youtube_url),
        _subtype="plain",
        _charset="utf-8",
    )
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "codex AI트레이딩 내용 전달_윤태준"

    with smtplib.SMTP_SSL("smtp.naver.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

    print("[SUCCESS] Email sent with GitHub-rendered report link.")


if __name__ == "__main__":
    send_professor_report()
