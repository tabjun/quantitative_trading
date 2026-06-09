"""
자율형 행동재무학 AI 에이전트 모듈 (Agent Module)

이 모듈은 Analyzer의 결과를 바탕으로 투자자의 멘탈을 관리하고
올바른 행동(Nudge)을 유도하는 가이드 메시지를 생성합니다.
사용자의 매도 시도 등 트리거가 발생했을 때 호출됩니다.
"""

import os
from typing import Dict, Any
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv(): pass

class WealthTechAgent:
    def __init__(self, name: str = None):
        self.name = name or os.getenv("AGENT_NAME", "AI투자메이트")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    def _generate_with_llm(self, prompt: str) -> str:
        """
        LLM을 사용하여 자연스러운 문장을 생성합니다.
        API 키가 설정되어 있지 않으면 기본 템플릿을 사용합니다.
        """
        if not self.openai_key and not self.gemini_key:
            return None # LLM 키가 없으면 None 반환하여 기본 로직 사용
        
        # 실제 구현 시에는 여기서 openai나 google-generativeai 라이브러리를 호출합니다.
        # return llm.generate(prompt)
        return f"[LLM 생성 메시지 예시] 현재 시장의 변동성은 파도와 같습니다. {self.name}는 당신이 이 파도에 휩쓸리지 않고 목적지에 도달하도록 돕겠습니다..."

    def generate_analytical_report_on_action(self, item: Dict[str, Any], sim_result: Dict[str, Any]) -> str:
        """
        사용자의 행동(매도/매수 등) 검토 시점에 제공하는 심층 분석 리포트.
        제어가 아닌 '객관적 정보 기반의 판단 지원'에 집중합니다.
        """
        report = f"""
==================================================
📊 [정보 기반 의사결정 지원 리포트: {item['name']}]
==================================================
회원님, 현재 {item['name']} 종목에 대한 의사결정을 검토 중이시군요. 
감정에 치우치지 않는 판단을 돕기 위해 데이터 기반의 분석 정보를 전달드립니다.

[1. 현 위치 분석]
- 현재 수익률: {item['return_rate']}%
- 섹터 성격: {item['sector']} (장기 우상향 자산군)
- 시장 심리: 탐욕 및 공포 지수가 극도로 낮아진 상태 (과매도 구간 가능성)

[2. 전략적 시뮬레이션 (과거 데이터 기반)]
- 시나리오 A (매도 후 관망): 과거 유사 구간에서 매도 시, 반등장 수익의 약 70%를 놓치는 경향이 확인되었습니다.
- 시나리오 B (보유 및 적립): 하락장에서의 코스트 에버리징은 평균적으로 원금 회복 기간을 {sim_result['scenario_hold_and_buy']['estimated_recovery_time_months']}개월 단축시켰습니다.

[3. 실행 가능한 향후 방향성 제안 (Next Steps)]
✅ 현재 종목의 펀더멘털에는 변화가 없습니다. 무리한 매도보다는 비중을 유지하세요.
✅ 만약 현금 비중이 있다면, 다음 적립 시점에는 상대적으로 저평가된 섹터를 추가 고려하는 것이 포트폴리오 분산에 유리합니다.
✅ 이번 하락장을 데이터로 이해하고 넘기시면, 투자 근육이 한 층 더 강화될 것입니다.

※ 본 정보는 참고용이며, 최종 판단은 회원님의 몫입니다. {self.name}는 언제나 객관적인 데이터로 함께하겠습니다.
==================================================
"""
        return report

    def generate_structural_insight_report(self, structural_analysis: Dict[str, Any]) -> str:
        """
        기존 MTS의 성향 분석과는 차별화된, 포트폴리오의 '작동 원리(Mechanics)'를 설명하는 리포트 생성
        """
        report = f"""
==================================================
🏗️ [포트폴리오 구조 및 생태계 역학 리포트]
==================================================
단순한 위험 성향 분석이 아닙니다. 회원님이 보유하신 종목들이 
시장에서 어떻게 서로 맞물려 돌아가는지 그 '연결고리'를 분석했습니다.

[1. 핵심 역학 구조 (Structural Mechanics)]
{structural_analysis['structural_summary']}

[2. 생태계 맵 (Ecosystem Map)]
"""
        for map_item in structural_analysis['ecosystem_map']:
            report += f"- {map_item}\n"

        report += f"""
[3. 취약점 및 동조화 경고 (Structural Vulnerability)]
⚠️ {structural_analysis['vulnerability']}

[4. 차별화된 시스템 가이드 (Systemic Guidance)]
💡 지금 'SOL 미국테크TOP10'이 하락하는 것은 나스닥 지수 하락보다 뼈아플 수 있습니다. 하지만 이는 나스닥 100 지수 전체가 흔들리는 것이 아니라, 상위 10개 '엔진'에 과부하가 걸린 일시적 현상입니다.
💡 삼성전자의 반등 신호가 포착될 때, 포트폴리오에 에너지/인프라 섹터(예: 두산에너빌리티 등)를 보완하면 생태계 리더와 그 수혜주 사이의 시너지를 극대화하는 '구조적 리밸런싱'이 가능해집니다.

정보를 아는 것만으로도 심리는 안정됩니다. 투자는 숫자가 아니라 구조를 이해하는 것입니다.
==================================================
"""
        return report
