"""
MTS API 연동 유틸리티 모듈 (MTS API Utility Module)

이 모듈은 실제 증권사 API(MTS/HTS)와의 통신을 캡슐화합니다.
현재는 프로젝트 초기 단계로, API 호출을 모방(Mocking)하는 형태로 구현되어 있습니다.
외부 모듈에서는 이 모듈의 인터페이스만 호출하여 결합도를 낮춥니다.
"""

from typing import Dict, List, Any
import time

class MtsAPIClient:
    def __init__(self, api_key: str, secret_key: str):
        """
        MTS API 클라이언트 초기화
        :param api_key: 증권사 API Key
        :param secret_key: 증권사 Secret Key
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.is_connected = False
        
    def connect(self) -> bool:
        """
        MTS 서버에 연결 (세션 획득)
        """
        print("[MTS API] 서버에 연결 중...")
        time.sleep(0.5)
        self.is_connected = True
        print("[MTS API] 연결 성공")
        return self.is_connected

    def get_portfolio(self, account_id: str) -> List[Dict[str, Any]]:
        """
        특정 계좌의 현재 보유 종목 리스트를 조회합니다.
        :param account_id: 계좌 번호
        :return: 보유 종목 리스트 (종목코드, 종목명, 수익률, 비중 등)
        """
        if not self.is_connected:
            raise ConnectionError("MTS API가 연결되지 않았습니다.")
        
        # Mocking Portfolio Data
        print(f"[MTS API] 계좌 {account_id} 포트폴리오 조회 중...")
        return [
            {
                "ticker": "466920",
                "name": "TIGER 미국테크TOP10 INDXX",
                "quantity": 150,
                "avg_price": 12000,
                "current_price": 10500,
                "return_rate": -12.5, # -12.5% 손실 중
                "sector": "US Tech",
                "is_etf": True
            },
            {
                "ticker": "005930",
                "name": "삼성전자",
                "quantity": 50,
                "avg_price": 80000,
                "current_price": 75000,
                "return_rate": -6.25,
                "sector": "Semiconductor",
                "is_etf": False
            }
        ]

    def get_market_condition(self) -> Dict[str, Any]:
        """
        현재 시장의 상태(지수, 변동성 등)를 조회합니다.
        """
        # Mocking Market Data
        return {
            "KOSPI": 2600.5,
            "NASDAQ": 15000.2,
            "market_status": "HIGH_VOLATILITY", # 급등락 장세
            "fear_and_greed_index": 25 # 극도의 공포
        }
