"""
Supervisor Module (Main Orchestrator)

이 스크립트는 전체 다중 에이전트(Multi-Agent) 아키텍처를 조율하는 Supervisor 역할을 합니다.
사용자의 요청(하락장에 대한 불안)이 발생하면, 하위 에이전트들을 순차적으로 호출하여
상황 분석 -> 과거 사례 매칭 -> 딥다이브 리포트 생성 파이프라인을 실행합니다.
"""

import time
from database.case_db import CaseDatabase
from agents.situation_analyzer import SituationAnalyzer
from agents.similarity_matcher import SimilarityMatcher
from agents.report_generator import ReportGenerator

def run_supervisor_pipeline(ticker: str, user_query: str):
    print(f"\n[Supervisor] '{ticker}' 종목 관련 사용자 문의 접수: \"{user_query}\"")
    print("-" * 60)
    time.sleep(1)

    # 1. DB 및 하위 에이전트 초기화
    case_db = CaseDatabase()
    analyzer_agent = SituationAnalyzer()
    matcher_agent = SimilarityMatcher(case_db)
    reporter_agent = ReportGenerator(name="AI투자메이트")

    # 2. 파이프라인 실행
    
    # Step 1: 상황 분석 (가상의 최근 뉴스 키워드 주입)
    # 실제로는 user_query나 뉴스 크롤러에서 추출된 키워드가 들어감
    current_keywords = ["전쟁", "중동", "유가", "지정학적 리스크", "폭격"]
    current_situation = analyzer_agent.analyze_current_situation(ticker, current_keywords)
    time.sleep(1)

    # Step 2: DTW & NLP 기반 과거 사례 매칭
    best_historical_match = matcher_agent.find_best_match(current_situation)
    time.sleep(1)

    # Step 3: 최종 심층 리포트 생성
    print("[Supervisor] 결과 취합 및 최종 리포트 생성 중...\n")
    time.sleep(1)
    final_report = reporter_agent.generate_historical_insight_report(current_situation, best_historical_match)
    
    print(final_report)

def main():
    print("🚀 AI 웰스테크 Multi-Agent 시스템 시작 (DTW & NLP 매칭 모드)")
    
    ticker = "TIGER 미국테크TOP10 INDXX"
    user_query = "아니 시발 내 주식 왜 떨어지는거야? 이스라엘 이란 뭐시기 터졌다는데 이거 계속 들고가도 돼?"
    
    run_supervisor_pipeline(ticker, user_query)

if __name__ == "__main__":
    main()
