"""2국면 레짐 전환 GARCH(1,1) — Gray(1996)/Klaassen(2002) collapsing 근사.

경로 의존성(regime path가 t개면 2^t가지) 문제를 Gray의 축약(collapsing) 절차로 피한다:
각 국면의 GARCH 재귀가 국면별 과거 h가 아니라, 직전 시점의 "필터링된(=t-1 정보까지만 쓴)"
집계 조건부분산 h_{t-1|t-1} = sum_i xi_{i,t-1|t-1} * h_{i,t-1} 를 공통으로 물려받는다.

**리포트/스코어링용으로 저장하는 값은 반드시 t시점 관측치를 보기 '전' 예측**
(h_pred[t] = xi_pred . h_i, 예측확률 xi_pred만 사용)이어야 한다. t시점 관측 x_t로
갱신한 필터링확률 xi_filt로 가중한 h_agg는 x_t 정보를 역으로 반영하므로, 그 값을
t시점 예측치로 쓰면 미래정보 유출이 된다(첫 버전의 버그 — 이후 xi_filt·h_agg는
오직 t+1 예측을 위한 내부 상태로만 쓴다).

국면별 조건부밀도는 t분포(GARCH-t와 동일 계열로 맞춤). 파라미터:
  omega_i, alpha_i, beta_i (i=1,2 저/고변동 국면), p11, p22(잔류확률), nu(공통 자유도)

내부 루프는 순수 파이썬 스칼라로 짠다 — n~10만 지점 x 최적화 반복 수천 회에서
numpy 스칼라 배열 오버헤드가 지배적이라, math 모듈 스칼라 연산이 훨씬 빠르다
(프로파일 결과 n=74000 1회 통과에 0.08초 — 최적화 반복에 충분히 빠르다).
"""
import math
import numpy as np
from scipy.optimize import minimize


def _unpack(theta):
    o1, a1, b1, o2, a2, b2, lp11, lp22, lnu = theta
    omega = (math.exp(o1), math.exp(o2))
    a1s = 1 / (1 + math.exp(-a1)) * 0.3
    a2s = 1 / (1 + math.exp(-a2)) * 0.3
    b1s = 1 / (1 + math.exp(-b1)) * (0.999 - a1s)
    b2s = 1 / (1 + math.exp(-b2)) * (0.999 - a2s)
    p11 = 1 / (1 + math.exp(-lp11)); p22 = 1 / (1 + math.exp(-lp22))
    nu = 2.05 + math.exp(lnu)
    return omega, (a1s, a2s), (b1s, b2s), p11, p22, nu


def _filter(eps_signed, theta, loglik_upto=None):
    """Hamilton 필터 + Gray 축약을 eps_signed 전 구간에 대해 한 번에 돌린다.

    loglik_upto: 이 인덱스 이전(<)까지만 로그우도에 합산한다(None이면 전체).
    학습 구간만으로 파라미터를 추정할 때 쓴다 — 검증 구간 관측치도 필터의 '예측 입력'
    으로는 계속 흘러가지만(실제 과거 수익률을 쓰는 1스텝 재귀는 GARCH-t와 동일 관례),
    그 구간의 가능도는 추정에 포함하지 않는다.

    반환 info["h_pred"][t] = t시점을 보기 전 예측한 집계 조건부분산(무유출).
    """
    (o1, o2), (a1, a2), (b1, b2), p11, p22, nu = _unpack(theta)
    n = len(eps_signed)
    denom = 2 - p11 - p22
    pi1 = (1 - p22) / denom if abs(denom) > 1e-8 else 0.5
    pi2 = 1 - pi1
    xi1, xi2 = pi1, pi2                       # 필터링확률 상태(다음 스텝 예측용)
    h_agg_state = float(np.var(eps_signed[:min(n, 2000)]) or 1.0)  # 필터링 집계분산 상태
    nu_const = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    loglik = 0.0
    upto = n if loglik_upto is None else loglik_upto
    h_pred_path = np.empty(n)
    xi_pred_path = np.empty((n, 2))
    prev_eps2 = h_agg_state
    exp_ = math.exp; log_ = math.log; log1p_ = math.log1p
    for t in range(n):
        if t == 0:
            xi1_pred, xi2_pred = pi1, pi2
        else:
            xi1_pred = p11 * xi1 + (1 - p22) * xi2
            xi2_pred = (1 - p11) * xi1 + p22 * xi2
        e2 = prev_eps2
        h1 = o1 + a1 * e2 + b1 * h_agg_state
        h2 = o2 + a2 * e2 + b2 * h_agg_state
        if h1 < 1e-12: h1 = 1e-12
        if h2 < 1e-12: h2 = 1e-12
        h_pred_path[t] = xi1_pred * h1 + xi2_pred * h2     # 유출 없는 1스텝 예측
        xi_pred_path[t, 0] = xi1_pred; xi_pred_path[t, 1] = xi2_pred

        x = eps_signed[t]; x2 = x * x
        if t < upto:
            lf1 = nu_const - 0.5 * log_(h1 * (nu - 2) / nu) - (nu + 1) / 2 * log1p_(x2 * nu / (h1 * (nu - 2) ** 2))
            lf2 = nu_const - 0.5 * log_(h2 * (nu - 2) / nu) - (nu + 1) / 2 * log1p_(x2 * nu / (h2 * (nu - 2) ** 2))
            if lf1 > 700: lf1 = 700.0
            elif lf1 < -700: lf1 = -700.0
            if lf2 > 700: lf2 = 700.0
            elif lf2 < -700: lf2 = -700.0
            w1 = xi1_pred * exp_(lf1); w2 = xi2_pred * exp_(lf2)
            s = w1 + w2
            if not math.isfinite(s) or s <= 0:
                return -1e10, None
            loglik += log_(s)
            xi1 = w1 / s; xi2 = w2 / s
        else:
            # 검증 구간: 우도에는 안 넣지만, 국면확률은 실제 관측으로 계속 갱신한다
            # (그래야 재귀가 학습 종료 시점에 멈추지 않고 실제 시장 정보를 계속 반영한다)
            lf1 = nu_const - 0.5 * log_(h1 * (nu - 2) / nu) - (nu + 1) / 2 * log1p_(x2 * nu / (h1 * (nu - 2) ** 2))
            lf2 = nu_const - 0.5 * log_(h2 * (nu - 2) / nu) - (nu + 1) / 2 * log1p_(x2 * nu / (h2 * (nu - 2) ** 2))
            lf1 = min(700.0, max(-700.0, lf1)); lf2 = min(700.0, max(-700.0, lf2))
            w1 = xi1_pred * exp_(lf1); w2 = xi2_pred * exp_(lf2)
            s = w1 + w2
            xi1, xi2 = (w1 / s, w2 / s) if (math.isfinite(s) and s > 0) else (xi1_pred, xi2_pred)
        h_agg_state = xi1 * h1 + xi2 * h2
        prev_eps2 = x2
    return loglik, dict(h_pred=h_pred_path, xi_pred=xi_pred_path,
                         omega=(o1, o2), alpha=(a1, a2), beta=(b1, b2), p11=p11, p22=p22, nu=nu)


def fit_ms_garch(r_pct, split_idx, n_restarts=2, maxiter=1200, max_fit_n=None, seed=0):
    """train 구간(< split_idx)의 로그우도만으로 MLE, 이후 전 구간 1스텝 예측을 함께 반환한다.

    r_pct: 수익률*100(GARCH-t와 동일 스케일), 원 인덱스 전체 길이(train+test).
    max_fit_n: 최적화(반복 수백~수천 회) 비용을 줄이려면 학습구간 중 최근 max_fit_n개
    연속 구간만으로 파라미터를 추정한다(None이면 학습구간 전체). 최종 1스텝 예측은
    항상 전체 구간(_filter 1회 통과)으로 다시 계산하므로 예측 자체가 잘리지는 않는다.
    """
    fit_r = r_pct[:split_idx]
    if max_fit_n is not None and split_idx > max_fit_n:
        fit_r = r_pct[split_idx - max_fit_n: split_idx]
    inits = [
        [-2.0, -1.0, 1.0, -1.0, 0.5, 1.5, 2.0, 2.0, 1.0],
        [-2.5, -0.5, 0.5, -1.5, 0.0, 1.0, 1.5, 1.5, 1.0],
        [-3.0, 0.0, 2.0, -0.5, 1.0, 0.5, 2.5, 1.0, 0.5],
    ][:n_restarts]
    best = None
    for x0 in inits:
        try:
            res = minimize(lambda th: -_filter(fit_r, th)[0], x0, method="Nelder-Mead",
                            options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-3))
            ll = -res.fun
            if best is None or ll > best[0]:
                best = (ll, res.x)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("MS-GARCH 수렴 실패")
    ll, theta = best
    _, info_full = _filter(r_pct, theta, loglik_upto=split_idx)   # 전 구간 1스텝 예측(무유출)
    return theta, ll, info_full
