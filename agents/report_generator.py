"""
Agent 3: 리포트 생성 에이전트 (Report Generator)

상황 분석 데이터와 매칭된 과거 사례 데이터를 결합하여
투자자에게 구조적 이유와 행동 지침을 제시하는 심층 리포트를 작성합니다.
"""

from typing import Dict, Any

class ReportGenerator:
    def __init__(self, name: str = "AI투자메이트"):
        self.name = name

    def generate_historical_insight_report(self, current: Dict[str, Any], historical: Dict[str, Any]) -> str:
        """
        과거 사례를 기반으로 현재 하락의 원인과 향후 전망을 설명하는 리포트 생성
        """
        dtw_score = historical['match_metrics']['dtw_distance']
        nlp_score = historical['match_metrics']['nlp_similarity_score']
        
        # DTW 거리가 낮을수록, NLP 점수가 높을수록 매칭률이 높다고 판단
        match_confidence = min(99, int((1 - (dtw_score / 10.0) + nlp_score) * 100))
        
        report = f"""
==================================================
📜 [과거 데이터 기반 딥다이브 리포트]
==================================================
"왜 내 주식만 떨어질까?"라는 불안감, {self.name}가 데이터로 답해드립니다.
현재 하락은 개별 종목의 문제가 아닌, 거시적 '역사적 패턴'의 반복일 확률이 높습니다.

[1. 현재 시장 상황 분석]
- 주요 원인: {current['macro_context']}
- 현재 누적 하락률: {current['current_drop']}

[2. 🔍 가장 유사한 과거 사례 매칭 (DTW & NLP 분석)]
AI가 수십 년간의 방대한 증시 데이터를 DTW(시계열 패턴)와 텍스트 유사도로 분석한 결과, 
현재 흐름은 아래 과거 사례와 약 {match_confidence}% 유사성을 보입니다.

▶ 매칭 사례: {historical['event_name']}
- 당시 최대 하락률: {historical['max_drop']} (하락 기간: 약 {historical['duration_months']}개월)
- 하락의 본질적 이유: {', '.join(historical['context_keywords'])}
- 차트 패턴(DTW 거리): {dtw_score} (낮을수록 현재 하락 궤적과 똑같음을 의미)

[3. 💡 반등의 역사 및 트리거 (Recovery Mechanics)]
당시 시장은 영원히 하락하지 않았습니다. 반등을 이끈 핵심 트리거는 다음과 같았습니다.
👉 반등 트리거: {historical['recovery_trigger']}
👉 전고점 회복 소요 시간: 약 {historical['recovery_time_months']}개월

[4. 🚀 Actionable Guide (우리의 전략)]
과거 {historical['event_name']} 당시, 공포에 질려 매도한 투자자들은 이후 찾아온 V자 반등을 놓쳤습니다.
현재의 지정학적 리스크 역시 펀더멘털을 파괴하는 이슈가 아닌 '시스템적 노이즈'에 가깝습니다.
현재는 매도할 때가 아니라, 과거 사례의 '반등 트리거'가 뉴스에 등장할 때까지 비중을 유지하며 기다려야 하는 구간입니다.

{self.name}는 감정이 아닌 '역사적 데이터'로 회원님의 자산을 지킵니다.
==================================================
"""
        return report
