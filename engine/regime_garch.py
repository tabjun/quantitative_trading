"""레짐 전환 GARCH 2종 — MS-GARCH(마르코프 전환)와 TAR-GARCH(임계값 전환).

23번 예비검증(`test/results/23_model_comparison_20260908/regime_switching_prelim/`)에서
쓰던 구현을 24번 정식 통합에 맞춰 `engine/` 공용 패키지로 올린 것이다. 예비검증 쪽
파일은 그 회차의 기록이라 손대지 않는다(AGENTS.md 2.3 완료 실험 read-only).

두 모델의 차이:
  MS-GARCH  — 국면이 관측되지 않는다. 마르코프 체인으로 국면 확률을 추정한다(Hamilton 필터).
  TAR-GARCH — 관측 가능한 임계변수가 문턱값을 넘는지로 국면이 즉시 결정된다. 필터링이 없다.

둘 다 국면별 조건부밀도는 t분포를 쓴다(GARCH-t와 같은 계열로 맞춰 비교 가능하게 한다).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

# ─────────────────────────────────────────────────────────────────────────────
# MS-GARCH — Gray(1996)/Klaassen(2002) collapsing 근사
#
# 경로 의존성(regime path가 t개면 2^t가지)을 Gray의 축약 절차로 피한다: 각 국면의 GARCH
# 재귀가 국면별 과거 h가 아니라, 직전 시점의 "필터링된(=t-1 정보까지만 쓴)" 집계 조건부분산
# h_{t-1|t-1} = Σ_i ξ_{i,t-1|t-1} h_{i,t-1} 를 공통으로 물려받는다.
#
# 저장하는 예측값은 반드시 **t시점 관측치를 보기 전** 예측(h_pred[t] = ξ_pred · h_i)이어야
# 한다. t시점 관측 x_t로 갱신한 필터링확률 ξ_filt로 가중하면 x_t가 역유입돼 미래정보 유출이
# 된다(예비검증 첫 버전의 버그 — 이후 ξ_filt·h_agg는 t+1 예측용 내부 상태로만 쓴다).
#
# 내부 루프를 순수 파이썬 스칼라로 짠 이유: n~10만 지점 × 최적화 반복 수천 회에서는
# numpy 스칼라 배열 오버헤드가 지배적이라 math 모듈 스칼라 연산이 훨씬 빠르다.
# ─────────────────────────────────────────────────────────────────────────────

_MS_INITS = [
    [-2.0, -1.0, 1.0, -1.0, 0.5, 1.5, 2.0, 2.0, 1.0],
    [-2.5, -0.5, 0.5, -1.5, 0.0, 1.0, 1.5, 1.5, 1.0],
    [-3.0, 0.0, 2.0, -0.5, 1.0, 0.5, 2.5, 1.0, 0.5],
    [-1.5, -1.5, 1.5, -2.0, 0.2, 2.0, 3.0, 2.5, 1.5],
    [-3.5, 0.5, 1.0, -1.0, 1.5, 1.0, 1.0, 3.0, 0.2],
]


def _ms_unpack(theta):
    o1, a1, b1, o2, a2, b2, lp11, lp22, lnu = theta
    omega = (math.exp(o1), math.exp(o2))
    a1s = 1 / (1 + math.exp(-a1)) * 0.3
    a2s = 1 / (1 + math.exp(-a2)) * 0.3
    b1s = 1 / (1 + math.exp(-b1)) * (0.999 - a1s)
    b2s = 1 / (1 + math.exp(-b2)) * (0.999 - a2s)
    p11 = 1 / (1 + math.exp(-lp11))
    p22 = 1 / (1 + math.exp(-lp22))
    nu = 2.05 + math.exp(lnu)
    return omega, (a1s, a2s), (b1s, b2s), p11, p22, nu


def ms_filter(eps_signed, theta, loglik_upto=None):
    """Hamilton 필터 + Gray 축약을 전 구간에 한 번에 돌린다.

    loglik_upto: 이 인덱스 이전(<)까지만 로그우도에 합산한다(None이면 전체). 검증 구간
    관측치도 필터의 예측 입력으로는 계속 흘러가지만(실제 과거 수익률로 1스텝 재귀하는
    것은 GARCH-t와 동일 관례), 그 구간의 가능도는 추정에 넣지 않는다.
    """
    (o1, o2), (a1, a2), (b1, b2), p11, p22, nu = _ms_unpack(theta)
    n = len(eps_signed)
    denom = 2 - p11 - p22
    pi1 = (1 - p22) / denom if abs(denom) > 1e-8 else 0.5
    pi2 = 1 - pi1
    xi1, xi2 = pi1, pi2
    h_agg_state = float(np.var(eps_signed[:min(n, 2000)]) or 1.0)
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
        h_pred_path[t] = xi1_pred * h1 + xi2_pred * h2
        xi_pred_path[t, 0] = xi1_pred; xi_pred_path[t, 1] = xi2_pred

        x = eps_signed[t]; x2 = x * x
        lf1 = nu_const - 0.5 * log_(h1 * (nu - 2) / nu) - (nu + 1) / 2 * log1p_(x2 * nu / (h1 * (nu - 2) ** 2))
        lf2 = nu_const - 0.5 * log_(h2 * (nu - 2) / nu) - (nu + 1) / 2 * log1p_(x2 * nu / (h2 * (nu - 2) ** 2))
        if lf1 > 700: lf1 = 700.0
        elif lf1 < -700: lf1 = -700.0
        if lf2 > 700: lf2 = 700.0
        elif lf2 < -700: lf2 = -700.0
        w1 = xi1_pred * exp_(lf1); w2 = xi2_pred * exp_(lf2)
        s = w1 + w2
        if t < upto:
            if not math.isfinite(s) or s <= 0:
                return -1e10, None
            loglik += log_(s)
            xi1 = w1 / s; xi2 = w2 / s
        else:
            xi1, xi2 = (w1 / s, w2 / s) if (math.isfinite(s) and s > 0) else (xi1_pred, xi2_pred)
        h_agg_state = xi1 * h1 + xi2 * h2
        prev_eps2 = x2
    return loglik, dict(h_pred=h_pred_path, xi_pred=xi_pred_path,
                        omega=(o1, o2), alpha=(a1, a2), beta=(b1, b2), p11=p11, p22=p22, nu=nu)


def fit_ms_garch(r_pct, split_idx, n_restarts=2, maxiter=1200, max_fit_n=None):
    """학습 구간(< split_idx) 로그우도만으로 MLE, 이후 전 구간 1스텝 예측을 함께 반환한다.

    r_pct    : 수익률×100(GARCH-t와 같은 스케일), 원 인덱스 전체 길이(train+test).
    max_fit_n: 최적화 비용을 줄이려면 학습구간 중 **최근** max_fit_n개 연속 구간만으로
               파라미터를 추정한다(None이면 학습구간 전체). 최종 1스텝 예측은 항상 전체
               구간을 다시 한 번 통과시켜 계산하므로 예측 자체가 잘리지는 않는다.
    """
    fit_r = r_pct[:split_idx]
    if max_fit_n is not None and split_idx > max_fit_n:
        fit_r = r_pct[split_idx - max_fit_n: split_idx]
    best = None
    for x0 in _MS_INITS[:n_restarts]:
        try:
            res = minimize(lambda th: -ms_filter(fit_r, th)[0], x0, method="Nelder-Mead",
                           options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-3))
            ll = -res.fun
            if best is None or ll > best[0]:
                best = (ll, res.x)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("MS-GARCH 수렴 실패")
    ll, theta = best
    _, info_full = ms_filter(r_pct, theta, loglik_upto=split_idx)
    return theta, ll, info_full


# ─────────────────────────────────────────────────────────────────────────────
# TAR-GARCH(SETAR-GARCH) — 관측 가능한 임계변수로 국면이 결정론적으로 전환
#
# 임계변수는 직전 h구간 실현변동성(과거만 사용, 1칸 밀어 t시점 예측에 t 이전 정보만
# 들어가게 한다). 문턱값 τ는 프로파일 우도로 고른다: 후보 분위수마다 GARCH 파라미터를
# MLE로 적합하고 학습구간 로그우도가 가장 큰 τ를 채택한다(Hansen 1996/2000 방식).
# ─────────────────────────────────────────────────────────────────────────────

_TAR_INITS = [
    [-2.0, -1.0, 1.0, -1.0, 0.5, 1.5, 1.0],
    [-2.5, -0.5, 0.5, -1.5, 0.0, 1.0, 1.0],
    [-3.0, 0.0, 1.5, -0.5, 1.0, 0.5, 0.5],
]


def _tar_unpack(theta):
    o1, a1, b1, o2, a2, b2, lnu = theta
    omega = (math.exp(o1), math.exp(o2))
    a1s = 1 / (1 + math.exp(-a1)) * 0.3
    a2s = 1 / (1 + math.exp(-a2)) * 0.3
    b1s = 1 / (1 + math.exp(-b1)) * (0.999 - a1s)
    b2s = 1 / (1 + math.exp(-b2)) * (0.999 - a2s)
    nu = 2.05 + math.exp(lnu)
    return omega, (a1s, a2s), (b1s, b2s), nu


def tar_filter(eps_signed, switch_lagged, tau, theta, loglik_upto=None):
    """switch_lagged[t]는 t시점 예측 이전 정보만 담아야 한다(1칸 밀린 과거 실현변동성)."""
    (o1, o2), (a1, a2), (b1, b2), nu = _tar_unpack(theta)
    n = len(eps_signed)
    h = float(np.var(eps_signed[:min(n, 2000)]) or 1.0)
    nu_const = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    loglik = 0.0
    upto = n if loglik_upto is None else loglik_upto
    h_path = np.empty(n)
    prev_eps2 = h
    log_ = math.log; log1p_ = math.log1p
    for t in range(n):
        if switch_lagged[t] > tau:
            o, a, b = o2, a2, b2
        else:
            o, a, b = o1, a1, b1
        h = o + a * prev_eps2 + b * h
        if h < 1e-12: h = 1e-12
        h_path[t] = h
        x = eps_signed[t]; x2 = x * x
        if t < upto:
            lf = nu_const - 0.5 * log_(h * (nu - 2) / nu) - (nu + 1) / 2 * log1p_(x2 * nu / (h * (nu - 2) ** 2))
            if not math.isfinite(lf):
                return -1e10, None
            loglik += lf
        prev_eps2 = x2
    return loglik, dict(h_pred=h_path, omega=(o1, o2), alpha=(a1, a2), beta=(b1, b2), nu=nu, tau=tau)


def fit_tar_garch(r_pct, switch_lagged, split_idx, tau_candidates, n_restarts=2, maxiter=500):
    """τ 후보마다 MLE 적합 후 학습구간 로그우도가 최대인 τ를 채택한다(프로파일 우도)."""
    eps_tr = r_pct[:split_idx]
    sw_tr = switch_lagged[:split_idx]
    best_overall = None
    for tau in tau_candidates:
        best = None
        for x0 in _TAR_INITS[:n_restarts]:
            try:
                res = minimize(lambda th: -tar_filter(eps_tr, sw_tr, tau, th)[0], x0,
                               method="Nelder-Mead", options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-3))
                ll = -res.fun
                if best is None or ll > best[0]:
                    best = (ll, res.x)
            except Exception:
                continue
        if best is not None and (best_overall is None or best[0] > best_overall[0]):
            best_overall = (best[0], best[1], tau)
    if best_overall is None:
        raise RuntimeError("TAR-GARCH 수렴 실패")
    ll, theta, tau = best_overall
    _, info_full = tar_filter(r_pct, switch_lagged, tau, theta, loglik_upto=split_idx)
    return theta, tau, ll, info_full
