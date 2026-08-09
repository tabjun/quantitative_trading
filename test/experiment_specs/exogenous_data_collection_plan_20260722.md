# 외생변수 수집 계획 — Scrapling 적용 + 기존 인프라 통합

작성일: 2026-07-22 · 브랜치: `stock` · 근거: 17번 EDA(가격만으론 방향 무신호), 영상(Scrapling)

> 사용자 공유 영상(youtu.be/hdEweGeZpuE): **Scrapling** — 70만 star 적응형 웹 스크래핑
> 프레임워크, Cloudflare Turnstile/Interstitial 자동 우회, 레이아웃 변경 시 요소 자동 재탐색
> (adaptive parsing). Python 3.10+, BSD-3, v0.4.11(2026-07) 활발히 유지. 설치·import 검증 완료.

## 1. 왜 지금 외생변수인가 (EDA 근거)

17번 EDA가 확정: 가격 파생변수만으로는 방향 신호가 사실상 없다(자기상관≈0, 최고 변수도 R²
미미). 방향 신호를 얻으려면 **가격 밖의 정보 채널**을 데이터에 더해야 한다(문헌: 방향 개선은
거의 항상 외생정보로 달성). 이것이 다음 단계이고, Scrapling은 그 **수집 도구**다.

## 2. 기존 인프라와의 관계 (중복 금지)

이미 있는 것:
- `contexts/text_context.py`: 구글뉴스 RSS(한국어) + 로컬 CSV → `text_features_15m` 마트(감성,
  이벤트 수, shock z-score, 토픽 count). 4·6~8번에서 만든 것.
- `pipelines/ingest_text_context.py`: 운영 수집 엔트리포인트.

**Scrapling의 역할(보강, 대체 아님)**: RSS/단순 requests로 **못 뚫는** 소스를 담당한다.
(a) Cloudflare로 막힌 사이트, (b) JS 렌더링 필요한 동적 페이지, (c) 레이아웃이 자주 바뀌어
파서가 깨지는 소스. 수집 결과는 기존 `text_context`의 `TextRecord` 스키마로 흘려보내
`text_features_15m` 마트를 그대로 재사용한다(새 마트 만들지 않음).

## 3. 수집 후보 채널 (방향 신호 가능성 순, 사용자 확인 필요)

| 채널 | 소스 예 | 수집 방식 | 방향 신호 기대 | API 우선? |
|---|---|---|---|---|
| 뉴스·감성(한국) | 코인니스·블록미디어·토큰포스트 | Scrapling(일부 Cloudflare) | 중 | 일부 RSS 있음 |
| 뉴스·감성(글로벌) | CoinDesk·Cointelegraph | Scrapling/RSS | 중 | RSS 있음 |
| 공포탐욕지수 | alternative.me | **공개 API**(스크래핑 불필요) | 중 | ✅ API |
| 거래소 공지(상장/폐지) | 업비트 공지 | Scrapling(이벤트 쇼크) | 이벤트성 高 | 일부 API |
| cross-asset | 바이낸스 USD-BTC, DXY, 나스닥 | **공개 API**(스크래핑 불필요) | 중(레짐) | ✅ API |
| 온체인 | 거래소 순유입, 활성주소 | 대부분 유료/API | 중 | API(유료 다수) |
| 검색·소셜 | 구글트렌드, X/레딧 | Scrapling(대부분 차단) | 저~중 | 제한적 |

**원칙**: API가 있으면 API 우선(스크래핑은 마지막 수단). Scrapling은 API 없고 requests로
막히는 소스에만. 공포탐욕지수·cross-asset은 API라 즉시 착수 가능, 뉴스·공지가 Scrapling 대상.

## 4. 단계 (실제 실행 전 사용자 확인 후)

1. **API 우선 채널 먼저**(스크래핑 리스크 0): 공포탐욕지수, 바이낸스 USD-BTC/펀딩, DXY·나스닥.
   → `text_features_15m` 확장 또는 새 `market_context_15m` 마트로 15분 정렬 적재.
2. **Scrapling 채널**: 대상 사이트 확정 → `scrapling install`로 브라우저 설치(무거움, 이때만) →
   `StealthyFetcher`로 뉴스·공지 수집 → 기존 text_context 감성/토픽 파이프라인에 연결.
3. **EDA 재실행(17번 확장)**: 새 외생변수를 S6(방향 신호)에 넣어, 가격 파생변수엔 없던 방향
   신호가 외생변수엔 있는지 상호정보량으로 확인. 있으면 16번 모델에 반영.

## 5. 리스크·주의

- **법·윤리**: robots.txt 준수(Scrapling 옵션 지원), 과도한 요청 금지, 개인정보 미수집.
- **재현성**: 스크래핑은 시점 의존이라 수집분을 DuckDB에 타임스탬프와 함께 원본 저장(재수집
  없이 재현 가능하게). look-ahead 방지(뉴스 timestamp ≤ 예측 시점).
- **무게**: 브라우저 자동화(StealthyFetcher)는 무겁다 — API로 되는 건 API로. 브라우저 설치는
  실제 스크래핑 대상이 확정된 뒤에만.

## 6. 지금 확인이 필요한 것 (사용자 결정)

- 어느 채널부터 갈지: (권장) API 채널(공포탐욕·cross-asset) 먼저 → 방향 신호 유무 빠르게 확인
  후 뉴스 스크래핑 확장. 아니면 뉴스부터?
- 뉴스 소스 우선순위(한국 vs 글로벌), 온체인 유료 API 사용 여부.
