"""Consistency check on the M2 Part A derivations. Throwaway; not library code.

Checks, for the d-dimensional exponential-kernel Hawkes process:

  1. the grouped recursion against the O(n^2) definition
  2. that grouping must pool ALL components: the per-event ("index-based") variant
     is wrong exactly when two components share a timestamp
  3. the d = 1 reduction, bit-identically against the M1 univariate recursion
  4. the analytic gradient against central differences
  5. the stationary mean intensity against its d = 1 closed form

    python3 docs/derivations/check_multivariate_derivation.py
"""
import math
import random


# ----------------------------------------------------------------- definition
def brute_force(events, mu, alpha, beta, T):
    """Direct transcription of the definition. No recursion, no simplification.

    lambda_i(t) = mu_i + sum_j sum_{t^j_k < t} alpha[i][j]*beta*exp(-beta*(t - t^j_k))
    Lambda_i(T) = mu_i*T + sum_j alpha[i][j] * sum_k (1 - exp(-beta*(T - t^j_k)))
    nll         = sum_i Lambda_i(T) - sum_i sum_k log lambda_i(t^i_k)
    """
    d = len(events)
    total = 0.0
    for i in range(d):
        total += mu[i] * T
        for j in range(d):
            for t in events[j]:
                total += alpha[i][j] * (1.0 - math.exp(-beta * (T - t)))
    for i in range(d):
        for tk in events[i]:
            lam = mu[i]
            for j in range(d):
                for t in events[j]:
                    if t < tk:                       # strict, on TIME (C3, C8)
                        lam += alpha[i][j] * beta * math.exp(-beta * (tk - t))
            total -= math.log(lam)
    return total


# ------------------------------------------------------- grouped recursion
def pooled_distinct_times(events):
    """Distinct times pooled across ALL components, with per-component counts."""
    d = len(events)
    times = sorted({t for component in events for t in component})
    counts = []
    for s in times:
        counts.append([sum(1 for t in events[j] if t == s) for j in range(d)])
    return times, counts


def recursive(events, mu, alpha, beta, T, want_gradient=False):
    """Grouped over pooled distinct times.

    Accumulation is deliberately PER EVENT, not `count * value`: the two differ in
    f64 (see the c=7 case in the header of the derivation), and per-event
    accumulation is what makes the d=1 reduction bit-identical to M1.
    """
    d = len(events)
    s, c = pooled_distinct_times(events)
    if not s:
        nll = sum(mu[i] * T for i in range(d))
        if not want_gradient:
            return nll
        zero = [[0.0] * d for _ in range(d)]
        return nll, ([T] * d, zero, 0.0)

    B = [0.0] * d
    Bp = [0.0] * d
    E = [0.0] * d
    dE = [0.0] * d
    log_term = 0.0
    g_mu = [0.0] * d
    g_alpha = [[0.0] * d for _ in range(d)]
    S = [[0.0] * d for _ in range(d)]

    for r, s_r in enumerate(s):
        if r > 0:
            gap = s_r - s[r - 1]
            gd = math.exp(-beta * gap)
            for m in range(d):
                advanced = gd * (B[m] + c[r - 1][m])
                Bp[m] = -gap * advanced + gd * Bp[m]
                B[m] = advanced

        window = T - s_r
        wd = math.exp(-beta * window)
        contribution = -math.expm1(-beta * window)
        for j in range(d):
            for _ in range(c[r][j]):                 # per event, not c * value
                E[j] += contribution
                dE[j] += window * wd

        for i in range(d):
            if not c[r][i]:
                continue
            lam = mu[i]
            for j in range(d):
                lam += (alpha[i][j] * beta) * B[j]   # (alpha*beta)*B, as in M1
            ln_lam = math.log(lam)
            for _ in range(c[r][i]):
                log_term += ln_lam
                if want_gradient:
                    g_mu[i] += 1.0 / lam
                    for j in range(d):
                        g_alpha[i][j] += (beta * B[j]) / lam
                    for j in range(d):
                        S[i][j] += (B[j] + beta * Bp[j]) / lam

    nll = sum(mu[i] * T for i in range(d))
    compensator = 0.0
    for i in range(d):
        for j in range(d):
            compensator += alpha[i][j] * E[j]
    nll = nll + compensator - log_term
    if not want_gradient:
        return nll

    grad_mu = [T - g_mu[i] for i in range(d)]
    grad_alpha = [[E[j] - g_alpha[i][j] for j in range(d)] for i in range(d)]
    grad_beta = 0.0
    for i in range(d):
        for j in range(d):
            grad_beta += alpha[i][j] * (dE[j] - S[i][j])
    return nll, (grad_mu, grad_alpha, grad_beta)


def recursive_per_event(events, mu, alpha, beta, T):
    """WRONG on cross-component ties: advances per EVENT rather than per distinct
    time, so at a shared timestamp the event processed first excites the other."""
    d = len(events)
    pooled = sorted((t, j) for j in range(d) for t in events[j])
    if not pooled:
        return sum(mu[i] * T for i in range(d))
    B = [0.0] * d
    log_term = 0.0
    E = [0.0] * d
    previous = pooled[0][0]
    pending = None
    for t, j in pooled:
        gap = t - previous
        gd = math.exp(-beta * gap)
        for m in range(d):
            B[m] *= gd
        if pending is not None:
            B[pending] += math.exp(-beta * gap)      # add the previous event now
        previous = t
        lam = mu[j] + beta * sum(alpha[j][m] * B[m] for m in range(d))
        log_term += math.log(lam)
        E[j] += -math.expm1(-beta * (T - t))
        pending = j
    nll = sum(mu[i] * T for i in range(d))
    nll += sum(alpha[i][j] * E[j] for i in range(d) for j in range(d))
    return nll - log_term


# ------------------------------------------------------ M1 univariate recursion
def univariate(times, mu, alpha, beta, T):
    if not times:
        return mu * T
    comp = log_term = 0.0
    B = 0.0
    prev_t, prev_c = times[0], 0
    for tk in times:
        if tk != prev_t:
            dgap = tk - prev_t
            B = math.exp(-beta * dgap) * (B + prev_c)
            prev_t, prev_c = tk, 0
        log_term += math.log(mu + alpha * beta * B)
        comp += -math.expm1(-beta * (T - tk))
        prev_c += 1
    return mu * T + alpha * comp - log_term


def univariate_gradient(times, mu, alpha, beta, T):
    """M1's gradient exactly as the shipped code computes it, including the
    association `alpha * (X - Y)` for d_beta."""
    if not times:
        return T, 0.0, 0.0
    comp = 0.0
    B = Bp = 0.0
    g_mu = g_alpha = g_beta_log = g_beta_comp = 0.0
    prev_t, prev_c = times[0], 0
    for tk in times:
        if tk != prev_t:
            gap = tk - prev_t
            gd = math.exp(-beta * gap)
            adv = gd * (B + prev_c)
            Bp = -gap * adv + gd * Bp
            B = adv
            prev_t, prev_c = tk, 0
        lam = mu + (alpha * beta) * B
        window = T - tk
        wd = math.exp(-beta * window)
        comp += -math.expm1(-beta * window)
        g_mu += 1.0 / lam
        g_alpha += (beta * B) / lam
        g_beta_comp += window * wd
        g_beta_log += (B + beta * Bp) / lam
        prev_c += 1
    return T - g_mu, comp - g_alpha, alpha * (g_beta_comp - g_beta_log)


def random_case(rng, d, n_max=25, T=8.0, allow_ties=False):
    mu = [rng.uniform(0.05, 2.0) for _ in range(d)]
    alpha = [[rng.uniform(0.0, 0.8 / d) for _ in range(d)] for _ in range(d)]
    beta = rng.uniform(0.2, 3.0)
    events = []
    for _ in range(d):
        n = rng.randint(0, n_max)
        if allow_ties:
            ts = sorted(float(rng.randint(0, 10)) for _ in range(n))
        else:
            ts = sorted(rng.uniform(0, T) for _ in range(n))
        events.append(ts)
    return events, mu, alpha, beta, T


rng = random.Random(20260823)

# 1. recursion vs definition ------------------------------------------------
worst = 0.0
for _ in range(600):
    d = rng.randint(1, 5)
    ev, mu, alpha, beta, T = random_case(rng, d, allow_ties=rng.random() < 0.5)
    a, b = recursive(ev, mu, alpha, beta, T), brute_force(ev, mu, alpha, beta, T)
    worst = max(worst, abs(a - b) / max(1.0, abs(b)))
print(f"1. recursion vs definition, worst relative : {worst:.3e}")

# 3. d = 1 reduction, bit-identical -----------------------------------------
mismatch = grad_mismatch = 0
for _ in range(600):
    ev, mu, alpha, beta, T = random_case(rng, 1, allow_ties=rng.random() < 0.5)
    m = recursive(ev, mu, alpha, beta, T)
    u = univariate(ev[0], mu[0], alpha[0][0], beta, T)
    if m.hex() != u.hex():
        mismatch += 1
    _, (gm, ga, gb) = recursive(ev, mu, alpha, beta, T, want_gradient=True)
    um, ua, ub = univariate_gradient(ev[0], mu[0], alpha[0][0], beta, T)
    if (gm[0].hex(), ga[0][0].hex(), gb.hex()) != (um.hex(), ua.hex(), ub.hex()):
        grad_mismatch += 1
print(f"3. d=1 vs M1 univariate nll, bit-identical : "
      f"{600 - mismatch}/600 exact, {mismatch} mismatched")
print(f"   d=1 gradient, bit-identical             : "
      f"{600 - grad_mismatch}/600 exact, {grad_mismatch} mismatched")

# 4. gradient vs central differences ----------------------------------------
def perturb(mu, alpha, beta, kind, idx, h):
    mu2 = list(mu); a2 = [row[:] for row in alpha]; b2 = beta
    if kind == "mu":
        mu2[idx] += h
    elif kind == "alpha":
        a2[idx[0]][idx[1]] += h
    else:
        b2 += h
    return mu2, a2, b2

worst_g = 0.0
for _ in range(120):
    d = rng.randint(1, 4)
    ev, mu, alpha, beta, T = random_case(rng, d, n_max=15,
                                         allow_ties=rng.random() < 0.5)
    if sum(len(e) for e in ev) == 0:
        continue
    _, (gm, ga, gb) = recursive(ev, mu, alpha, beta, T, want_gradient=True)
    h = 1e-6
    for i in range(d):
        up = perturb(mu, alpha, beta, "mu", i, h)
        dn = perturb(mu, alpha, beta, "mu", i, -h)
        num = (recursive(ev, *up, T) - recursive(ev, *dn, T)) / (2 * h)
        worst_g = max(worst_g, abs(gm[i] - num) / max(1.0, abs(num)))
        for j in range(d):
            up = perturb(mu, alpha, beta, "alpha", (i, j), h)
            dn = perturb(mu, alpha, beta, "alpha", (i, j), -h)
            num = (recursive(ev, *up, T) - recursive(ev, *dn, T)) / (2 * h)
            worst_g = max(worst_g, abs(ga[i][j] - num) / max(1.0, abs(num)))
    up = perturb(mu, alpha, beta, "beta", None, h)
    dn = perturb(mu, alpha, beta, "beta", None, -h)
    num = (recursive(ev, *up, T) - recursive(ev, *dn, T)) / (2 * h)
    worst_g = max(worst_g, abs(gb - num) / max(1.0, abs(num)))
print(f"4. gradient vs central differences, worst  : {worst_g:.3e}")

# 5. stationary mean intensity ----------------------------------------------
def stationary_mean(mu, alpha):
    d = len(mu)
    m = [[(1.0 if i == j else 0.0) - alpha[i][j] for j in range(d)] for i in range(d)]
    aug = [m[i][:] + [mu[i]] for i in range(d)]
    for col in range(d):
        p = max(range(col, d), key=lambda r: abs(aug[r][col]))
        aug[col], aug[p] = aug[p], aug[col]
        pivot = aug[col][col]
        for k in range(col, d + 1):
            aug[col][k] /= pivot
        for r in range(d):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                for k in range(col, d + 1):
                    aug[r][k] -= f * aug[col][k]
    return [aug[i][d] for i in range(d)]

lam = stationary_mean([0.6], [[0.5]])
print(f"5. stationary mean, d=1: {lam[0]!r}   mu/(1-alpha) = {0.6/0.5!r}")
lam2 = stationary_mean([0.2, 0.5], [[0.1, 0.6], [0.05, 0.15]])
print(f"   d=2: {lam2!r}")

# 2. cross-component tie counterexample --------------------------------------
print()
print("2. cross-component tie counterexample")
mu = [0.2, 0.5]
alpha = [[0.1, 0.6], [0.05, 0.15]]
beta, T = 1.2, 6.0
ev = [[1.0, 2.5], [2.5, 4.0]]
print(f"   d=2  mu={mu}  alpha={alpha}  beta={beta}  T={T}")
print(f"   events: node0={ev[0]}  node1={ev[1]}   (shared timestamp 2.5)")
print(f"   definition (brute force)           : {brute_force(ev, mu, alpha, beta, T)!r}")
print(f"   grouped over pooled distinct times : {recursive(ev, mu, alpha, beta, T)!r}")
print(f"   grouped per event (index-based)    : {recursive_per_event(ev, mu, alpha, beta, T)!r}  <- wrong")
no_tie = [[1.0, 2.5], [2.6, 4.0]]
print(f"   same data, tie removed (2.5 -> 2.6):")
print(f"     definition {brute_force(no_tie, mu, alpha, beta, T)!r}  "
      f"pooled {recursive(no_tie, mu, alpha, beta, T)!r}  "
      f"per-event {recursive_per_event(no_tie, mu, alpha, beta, T)!r}")
