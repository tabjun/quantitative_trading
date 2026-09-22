"""TAR-GARCH(SETAR-GARCH) — 관측 가능한 임계변수로 국면이 결정론적으로 전환되는 GARCH(1,1).

MS-GARCH(레짐전환)와의 차이: MS-GARCH는 국면이 숨겨져 있어 Hamilton 필터로 확률
추정이 필요하지만, TAR-GARCH는 임계변수(여기서는 직전 실현변동성)가 문턱값 τ를
넘는지로 국면이 **그 즉시 결정**된다. 필터링이 필요 없어 구현·적합이 훨씬 가볍다.

임계변수: 직전 h구간 실현변동성(과거만 사용, targets()의 pas를 1칸 밀어 t시점
예측에 t 이전 정보만 들어가게 한다 — leak 방지).

τ는 프로파일 우도(profile likelihood)로 고른다: 후보 분위수마다 GARCH 파라미터를
MLE로 적합하고, 학습구간 로그우도가 가장 큰 τ를 채택한다(Hansen 1996/2000 방식).
"""
import math
import numpy as np
from scipy.optimize import minimize


def _unpack(theta):
    o1, a1, b1, o2, a2, b2, lnu = theta
    omega = (math.exp(o1), math.exp(o2))
    a1s = 1 / (1 + math.exp(-a1)) * 0.3
    a2s = 1 / (1 + math.exp(-a2)) * 0.3
    b1s = 1 / (1 + math.exp(-b1)) * (0.999 - a1s)
    b2s = 1 / (1 + math.exp(-b2)) * (0.999 - a2s)
    nu = 2.05 + math.exp(lnu)
    return omega, (a1s, a2s), (b1s, b2s), nu


def _filter(eps_signed, switch_lagged, tau, theta, loglik_upto=None):
    """switch_lagged[t]는 t시점 예측 이전 정보만(과거 실현변동성, 1칸 밀린 값)."""
    (o1, o2), (a1, a2), (b1, b2), nu = _unpack(theta)
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
    """tau 후보마다 MLE 적합 후 학습구간 로그우도가 최대인 tau를 채택(프로파일 우도)."""
    inits = [
        [-2.0, -1.0, 1.0, -1.0, 0.5, 1.5, 1.0],
        [-2.5, -0.5, 0.5, -1.5, 0.0, 1.0, 1.0],
    ][:n_restarts]
    best_overall = None
    for tau in tau_candidates:
        sw_tr = switch_lagged[:split_idx]; eps_tr = r_pct[:split_idx]
        best = None
        for x0 in inits:
            try:
                res = minimize(lambda th: -_filter(eps_tr, sw_tr, tau, th)[0], x0,
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
    _, info_full = _filter(r_pct, switch_lagged, tau, theta, loglik_upto=split_idx)
    return theta, tau, ll, info_full
