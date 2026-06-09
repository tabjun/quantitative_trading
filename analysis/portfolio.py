"""
포트폴리오 및 행동재무학 데이터 분석 모듈 (Analyzer Module)

이 모듈은 MTS API에서 가져온 데이터를 바탕으로 
1) 보유 종목의 성격(섹터, 지수 추종 등)을 분석하고
2) 과거 백테스팅(타임머신 시각화) 계산을 수행합니다.
"""

from typing import Dict, List, Any

class PortfolioAnalyzer:
    def __init__(self):
        # 종목 간 관계성 및 생태계 데이터 (Ecosystem Mapping)
        self.ecosystem_data = {
            "466920": { # TIGER 미국테크TOP10
                "role": "Engine", # 지수의 핵심 엔진 역할
                "relation_to": "NASDAQ100",
                "characteristic": "나스닥100의 수익률을 견인하는 상위 10개 핵심 엔진. 지수 전체보다 변동성은 크지만 반등 시 탄력이 압도적임.",
                "synergy_with": ["Semiconductor", "AI Infrastructure"]
            },
            "005930": { # 삼성전자
                "role": "Ecosystem Leader", # 생태계 최상위 포식자
                "dependents": ["Semiconductor Equipment", "Electronic Parts", "Energy/Infrastructure"],
                "characteristic": "한국 증시의 심장. 삼성전자의 가동률은 두산에너빌리티(에너지), 원익IPS(장비) 등 밸류체인 전반의 방향타 역할을 함.",
                "cycle_influence": "반도체 업황 사이클"
            }
        }

    def analyze_structural_relationships(self, portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        단순 수익률이 아닌, 종목 간의 '구조적 관계'와 '생태계 시너지'를 분석합니다.
        """
        analysis_result = {
            "structural_summary": "",
            "ecosystem_map": [],
            "vulnerability": ""
        }
        
        has_engine = False
        has_leader = False
        
        for item in portfolio:
            data = self.ecosystem_data.get(item['ticker'])
            if not data: continue
            
            if data['role'] == "Engine":
                has_engine = True
                analysis_result['ecosystem_map'].append(
                    f"🚀 [엔진 확인] {item['name']}은(는) 시장의 수익률을 끌어올리는 상위 10대 핵심 엔진입니다. "
                    f"전체 지수보다 무겁지만 힘(탄력)이 강한 상태입니다."
                )
            
            if data['role'] == "Ecosystem Leader":
                has_leader = True
                analysis_result['ecosystem_map'].append(
                    f"🏗️ [생태계 리더 확인] {item['name']}은(는) 공급망 전체를 리딩합니다. "
                    f"이 종목의 회복은 보유하신 다른 제조/장비 섹터의 동반 반등을 예고하는 선행 지표입니다."
                )
        
        # 구조적 결합 분석
        if has_engine and has_leader:
            analysis_result['structural_summary'] = "현재 포트폴리오는 '미국 테크 엔진'과 '한국 제조 리더'가 결합된 구조입니다."
            analysis_result['vulnerability'] = "두 축 모두 '금리'와 '수출 업황'이라는 동일한 외부 변수에 노출되어 있어 동조화 현상이 강할 수 있습니다."
            
        return analysis_result

    def simulate_time_machine(self, ticker: str, current_loss: float) -> Dict[str, Any]:
        """
        과거 유사한 하락장에서 '매도 후 저점 재매수(타이밍)' 전략과 
        '적립식 장기 보유(존버)' 전략의 결과를 비교 시뮬레이션 합니다.
        (현재는 하드코딩된 예시 데이터를 반환합니다)
        """
        # 예시: 2022년 하락장 당시 데이터를 기반으로 한 시뮬레이션 값
        return {
            "ticker": ticker,
            "scenario_sell_and_timing": {
                "name": "타이밍 매매 시도",
                "estimated_recovery_time_months": 36,
                "final_return_after_5_years": 15.2,
                "missed_best_days": 12 # 놓친 폭등일 수
            },
            "scenario_hold_and_buy": {
                "name": "하락장 무시하고 꾸준히 매수",
                "estimated_recovery_time_months": 14,
                "final_return_after_5_years": 85.4,
                "missed_best_days": 0
            },
            "conclusion": f"과거 데이터를 볼 때, 현재의 {current_loss}% 손실 구간에서 매도할 경우 폭등장을 놓쳐 장기 수익률이 크게 훼손될 확률이 높습니다."
        }
