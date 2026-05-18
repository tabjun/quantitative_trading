import os
import smtplib
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# .env 파일 로드 로직 (python-dotenv가 없어도 작동하도록 기본 파서 구현)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()

def send_professor_report():
    print("⏳ 알림 메일 발송 준비 중...")
    
    # 환경변수에서 자격 증명 가져오기
    sender_email = os.environ.get("NAVER_EMAIL_ID")
    sender_password = os.environ.get("NAVER_APP_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")
    
    if not sender_email or not sender_password or not receiver_email:
        print("❌ 오류: .env 파일에 이메일 정보(NAVER_EMAIL_ID, NAVER_APP_PASSWORD, RECEIVER_EMAIL)가 설정되지 않았습니다.")
        return

    smtp_server = "smtp.naver.com"
    smtp_port = 465
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "[윤태준 퀀트 트레이딩] 교수님 업비트 분석 및 학습 결과 보고서 전달드립니다."
    
    body = """안녕하세요 교수님

전달주신 시계열 알고리즘으로 단변량 분석 수행한 결과 정리해서 전달드립니다.
해당 메일은 gemini cli + 파이썬 자동화로 분석 - 보고서 생성 - 메일 전달까지 자동화해서 전달드리는 것입니다 하하. 

감사합니다.
윤태준 드림"""
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    # 1. MD 파일 첨부
    md_path = "analysis_report.md"
    if os.path.exists(md_path):
        with open(md_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(md_path)}")
            msg.attach(part)
        print(f"📎 {md_path} 첨부 완료.")
    else:
        # Try one level up or relative
        alt_md_path = os.path.join(os.path.dirname(__file__), "analysis_report.md")
        if os.path.exists(alt_md_path):
            with open(alt_md_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(alt_md_path)}")
                msg.attach(part)
            print(f"📎 {alt_md_path} 첨부 완료.")
        else:
            print(f"⚠️ 경고: 첨부할 마크다운 파일을 찾을 수 없습니다 ({md_path})")

    # 2. PDF 파일 첨부
    pdf_path = "analysis_report.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            part = MIMEBase("application", "pdf")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(pdf_path)}")
            msg.attach(part)
        print(f"📎 {pdf_path} 첨부 완료.")
    else:
        alt_pdf_path = os.path.join(os.path.dirname(__file__), "analysis_report.pdf")
        if os.path.exists(alt_pdf_path):
            with open(alt_pdf_path, "rb") as f:
                part = MIMEBase("application", "pdf")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(alt_pdf_path)}")
                msg.attach(part)
            print(f"📎 {alt_pdf_path} 첨부 완료.")
    
    try:
        print("⏳ 네이버 SMTP 서버에 연결 중...")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.close()
        print("💡 학습 완료 보고서 및 첨부파일 메일 발송 성공!")
    except Exception as e:
        print(f"❌ 메일 발송 실패: {e}")

if __name__ == "__main__":
    send_professor_report()