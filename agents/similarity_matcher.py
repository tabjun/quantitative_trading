"""
Agent 2: 유사도 매칭 에이전트 (Similarity Matcher)

시계열 데이터(DTW)와 텍스트 데이터(NLP)를 결합하여 
Case DB에서 현재 상황과 가장 유사한 역사적 사례를 찾아냅니다.
"""

from typing import Dict, Any, List
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class SimilarityMatcher:
    def __init__(self, db):
        self.db = db

    def calculate_dtw_distance(self, seq1: List[float], seq2: List[float]) -> float:
        """
        두 시계열 간의 형태적 유사도(DTW 거리)를 계산합니다.
        거리가 짧을수록 차트 패턴이 유사함을 의미합니다.
        """
        x = np.array(seq1)
        y = np.array(seq2)
        # fastdtw 반환값: (distance, path)
        distance, path = fastdtw(x, y, dist=euclidean)
        return distance

    def calculate_nlp_similarity(self, current_keywords: List[str], historical_keywords: List[str]) -> float:
        """
        현재 뉴스 키워드와 과거 이슈 키워드 간의 일치도를 계산합니다.
        (실제로는 SentenceTransformer나 OpenAI Embedding 코사인 유사도 사용)
        여기서는 간단한 자카드/교집합 기반 Mocking으로 대체합니다.
        """
        # "전쟁", "중동", "지정학적" 등의 키워드가 겹치는지 확인
        overlap = set(current_keywords).intersection(set(historical_keywords))
        return len(overlap) / (len(current_keywords) + 1e-5) # 점수가 높을수록 유사함

    def find_best_match(self, current_situation: Dict[str, Any]) -> Dict[str, Any]:
        """
        현재 상황과 가장 유사한 과거 사례를 도출합니다.
        """
        print("[Agent: SimilarityMatcher] Case DB와 DTW 차트 유사도 및 NLP 문맥 유사도 매칭 중...")
        
        cases = self.db.get_all_cases()
        best_case = None
        best_score = float('inf') # 종합 점수 (낮을수록 매칭됨)
        
        for case in cases:
            # 1. 시계열 유사도 계산
            dtw_dist = self.calculate_dtw_distance(
                current_situation["current_trend_sequence"], 
                case["price_sequence"]
            )
            
            # 2. 문맥 유사도 계산 (높을수록 좋으므로 점수에서 차감)
            nlp_sim = self.calculate_nlp_similarity(
                current_situation["news_keywords"], 
                case["context_keywords"]
            )
            
            # 3. 가중치 기반 결합 (DTW 70%, NLP 30%)
            # 임의의 스케일링 적용 (DTW는 낮을수록, NLP는 높을수록 좋음)
            combined_score = (dtw_dist * 0.7) - (nlp_sim * 10) 
            
            if combined_score < best_score:
                best_score = combined_score
                best_case = case
                best_case['match_metrics'] = {
                    "dtw_distance": round(dtw_dist, 2),
                    "nlp_similarity_score": round(nlp_sim, 2)
                }
                
        print(f"[Agent: SimilarityMatcher] 최적 매칭 사례 발견: {best_case['event_name']}")
        return best_case
