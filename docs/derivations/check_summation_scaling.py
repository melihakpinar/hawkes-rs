"""How does the recursion-vs-brute-force discrepancy scale with n?

Evidence for `univariate_loglikelihood.md` §5's statement that the Part B step 7
comparison gate is RELATIVE and bounded at n <= 50000. Throwaway reference code, not
`hawk` code: python3, no dependencies.

    python3 docs/derivations/check_summation_scaling.py


math.fsum is exactly rounded, so it serves as ground truth and lets each side's own
error be measured separately rather than only their difference.
"""
import math, random

def recursion(t, mu, alpha, beta, T):
    n = len(t)
    if n == 0: return mu * T
    comp = log_term = 0.0
    B = 0.0; prev_t = t[0]; prev_c = 0
    for tk in t:
        if tk != prev_t:
            d = tk - prev_t
            B = math.exp(-beta * d) * (B + prev_c)
            prev_t = tk; prev_c = 0
        log_term += math.log(mu + alpha * beta * B)
        comp += -math.expm1(-beta * (T - tk))
        prev_c += 1
    return mu * T + alpha * comp - log_term

def brute(t, mu, alpha, beta, T, exact=False):
    add = math.fsum if exact else sum
    comp = mu * T + alpha * add(-math.expm1(-beta * (T - x)) for x in t)
    logs = []
    for k, tk in enumerate(t):
        lam = mu + alpha * beta * add(math.exp(-beta * (tk - t[j])) for j in range(k))
        logs.append(math.log(lam))
    return comp - add(logs)

def kahan_brute(t, mu, alpha, beta, T):
    def ksum(it):
        s = c = 0.0
        for x in it:
            y = x - c; tmp = s + y; c = (tmp - s) - y; s = tmp
        return s
    comp = mu * T + alpha * ksum(-math.expm1(-beta * (T - x)) for x in t)
    logs = [math.log(mu + alpha * beta * ksum(math.exp(-beta * (tk - t[j]))
                                              for j in range(k)))
            for k, tk in enumerate(t)]
    return comp - ksum(logs)

random.seed(11)
print(f"{'n':>7s} {'|nll|':>12s} {'recur vs exact':>16s} {'naive vs exact':>16s} "
      f"{'kahan vs exact':>16s} {'recur vs naive':>16s}")
print("-" * 92)
for n in (100, 300, 1000, 3000, 6000):
    T = n / 2.0
    t = sorted(random.uniform(0, T) for _ in range(n))
    mu, alpha, beta = 1.3, 0.6, 0.9
    ex = brute(t, mu, alpha, beta, T, exact=True)
    r  = recursion(t, mu, alpha, beta, T)
    na = brute(t, mu, alpha, beta, T)
    ka = kahan_brute(t, mu, alpha, beta, T)
    rel = lambda a: abs(a - ex) / max(1.0, abs(ex))
    print(f"{n:7d} {abs(ex):12.2f} {rel(r):16.3e} {rel(na):16.3e} {rel(ka):16.3e} "
          f"{abs(r-na)/max(1.0,abs(ex)):16.3e}")
