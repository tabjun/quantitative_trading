"""
Case Database Module (과거 사례 데이터베이스)

과거 주식 시장에 큰 영향을 미쳤던 매크로 이벤트(지정학적 리스크, 금리, 전염병 등)와
당시의 주가 흐름(Time Series Sequence) 및 반등 트리거를 저장하는 Mock DB입니다.
"""

from typing import List, Dict, Any

class CaseDatabase:
    def __init__(self):
        # 과거 주요 폭락장 사례 
        self.cases: List[Dict[str, Any]] = [
            {
                "id": "CASE_2022_RATE_HIKE",
                "event_name": "2022년 미 연준 급진적 금리 인상",
                "sector": "Tech/Growth",
                "context_keywords": ["금리 인상", "인플레이션", "연준", "파월", "긴축"],
                # 정규화된 가격 흐름 (1.0에서 시작하여 하락 후 반등하는 패턴)
                "price_sequence": [1.0, 0.95, 0.88, 0.82, 0.75, 0.70, 0.68, 0.72, 0.78, 0.85],
                "max_drop": "-32%",
                "duration_months": 10,
                "recovery_trigger": "인플레이션(CPI) 지표 하락 확인 및 금리 인상 속도 조절 기대감",
                "recovery_time_months": 14
            },
            {
                "id": "CASE_2020_PANDEMIC",
                "event_name": "2020년 코로나19 팬데믹 쇼크",
                "sector": "All",
                "context_keywords": ["전염병", "팬데믹", "락다운", "봉쇄", "공급망 붕괴"],
                "price_sequence": [1.0, 0.90, 0.75, 0.65, 0.70, 0.80, 0.95, 1.05, 1.15],
                "max_drop": "-35%",
                "duration_months": 2,
                "recovery_trigger": "연준의 무제한 양적완화(QE) 및 백신 개발 가시화",
                "recovery_time_months": 5
            },
            {
                "id": "CASE_1990_GULF_WAR",
                "event_name": "1990년 걸프전 발발 및 유가 급등",
                "sector": "All",
                "context_keywords": ["전쟁", "중동", "지정학적 리스크", "유가 폭등", "침공"],
                "price_sequence": [1.0, 0.98, 0.92, 0.85, 0.80, 0.83, 0.89, 0.95, 1.02],
                "max_drop": "-20%",
                "duration_months": 3,
                "recovery_trigger": "다국적군의 빠르고 압도적인 승리 및 유가 안정화",
                "recovery_time_months": 6
            }
        ]

    def get_all_cases(self) -> List[Dict[str, Any]]:
        return self.cases
