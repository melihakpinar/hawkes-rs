"""Consistency check on the Part A derivations. Throwaway; not library code.

Checks the O(n) recursion (4.5) against the O(n^2) definition (3.3), and the
analytic gradient (G.1)/(G.2)/(G.7) against central differences -- including ties,
which is the case the textbook recursion (4.2) gets wrong.
"""
import math, random

def brute(t, mu, alpha, beta, T):                      # (3.3), the definition
    comp = mu * T + alpha * sum(1.0 - math.exp(-beta * (T - ti)) for ti in t)
    log_term = 0.0
    for tk in t:
        lam = mu + sum(alpha * beta * math.exp(-beta * (tk - ti)) for ti in t if ti < tk)
        log_term += math.log(lam)
    return comp - log_term

def recursive(t, mu, alpha, beta, T):                  # (4.5) + gradient, as coded in §5
    n = len(t)
    if n == 0:
        return mu * T, (T, 0.0, 0.0)
    comp_exc = log_term = d_mu_acc = d_alpha_acc = d_beta_comp = d_beta_log = 0.0
    B = Bp = 0.0
    prev_time, prev_count = t[0], 0
    for tk in t:
        if tk != prev_time:
            d = tk - prev_time
            dec = math.exp(-beta * d)
            B_new = dec * (B + prev_count)             # (4.4)
            Bp = -d * B_new + dec * Bp                 # (G.6), uses B_new
            B = B_new
            prev_time, prev_count = tk, 0
        lam = mu + alpha * beta * B
        w = T - tk
        dec_w = math.exp(-beta * w)
        log_term += math.log(lam)
        comp_exc += -math.expm1(-beta * w)
        d_mu_acc += 1.0 / lam
        d_alpha_acc += beta * B / lam
        d_beta_comp += w * dec_w
        d_beta_log += (B + beta * Bp) / lam
        prev_count += 1
    nll = mu * T + alpha * comp_exc - log_term
    return nll, (T - d_mu_acc, comp_exc - d_alpha_acc,
                 alpha * (d_beta_comp - d_beta_log))

def central(t, p, T, i, h=1e-6):
    up, dn = list(p), list(p)
    up[i] += h; dn[i] -= h
    return (recursive(t, *up, T)[0] - recursive(t, *dn, T)[0]) / (2 * h)

random.seed(7)
worst_val = worst_grad = 0.0
cases = []
# randomized distinct-timestamp cases
for _ in range(300):
    n = random.randint(0, 40)
    T = random.uniform(5.0, 50.0)
    t = sorted(random.uniform(0, T) for _ in range(n))
    cases.append((t, T, "random"))
# tie-heavy cases: the situation (4.2) gets wrong
for _ in range(150):
    T = random.uniform(5.0, 30.0)
    base = sorted(random.uniform(0, T) for _ in range(random.randint(1, 8)))
    t = sorted(x for x in base for _ in range(random.randint(1, 4)))
    cases.append((t, T, "ties"))
# degenerate cases
cases += [([], 10.0, "empty"), ([3.0], 10.0, "single"),
          ([2.0, 2.0, 2.0], 10.0, "all-tied"), ([0.0, 5.0, 10.0], 10.0, "endpoints")]

for t, T, kind in cases:
    mu = random.uniform(0.05, 3.0)
    alpha = random.uniform(0.01, 0.95)
    beta = random.uniform(0.1, 4.0)
    nll_r, grad = recursive(t, mu, alpha, beta, T)
    nll_b = brute(t, mu, alpha, beta, T)
    worst_val = max(worst_val, abs(nll_r - nll_b) / max(1.0, abs(nll_b)))
    if t:
        for i in range(3):
            num = central(t, (mu, alpha, beta), T, i)
            worst_grad = max(worst_grad,
                             abs(grad[i] - num) / max(1.0, abs(grad[i]), abs(num)))

print(f"cases: {len(cases)}")
print(f"worst relative |recursive - brute force| : {worst_val:.3e}")
print(f"worst relative |analytic - central diff| : {worst_grad:.3e}")

# Show that the textbook recursion (4.2) really is wrong on ties.
def recursive_naive(t, mu, alpha, beta, T):
    A, log_term = 0.0, 0.0
    for k, tk in enumerate(t):
        if k > 0:
            A = math.exp(-beta * (tk - t[k-1])) * (1.0 + A)   # (4.2)
        log_term += math.log(mu + alpha * beta * A)
    return mu*T + alpha*sum(1.0-math.exp(-beta*(T-ti)) for ti in t) - log_term

t, mu, alpha, beta, T = [1.0, 2.0, 2.0, 3.0], 0.7, 0.5, 1.3, 5.0
print(f"\ntie case t={t}")
print(f"  definition (3.3)      : {brute(t, mu, alpha, beta, T)!r}")
print(f"  grouped recursion(4.5): {recursive(t, mu, alpha, beta, T)[0]!r}")
print(f"  textbook (4.2)        : {recursive_naive(t, mu, alpha, beta, T)!r}  <- wrong")


# ---------------------------------------------------------------------------
# Cross-check against [Laub2015, eq. 21] under the reparametrization of
# univariate_loglikelihood.md §1.1.  A factor-of-beta error in that substitution
# would corrupt every downstream equation, so it is checked rather than asserted.
#
# Laub eq. 21, verbatim in Laub's own symbols (lambda, alpha_L, beta), horizon t_k:
#   l = sum_i log(lambda + alpha_L*A(i)) - lambda*t_k
#       + (alpha_L/beta) * sum_i [ exp(-beta*(t_k - t_i)) - 1 ]
# with A(i) = sum_{j<i} exp(-beta*(t_i - t_j)) and A(1) = 0  (eq. 20).
#
# Laub assumes a simple point process ("multiple arrivals cannot occur at the same
# time"), so this check uses distinct timestamps only -- eq. 20 is not claimed to
# hold under ties, which is the whole point of section 4.2.
def laub_eq21(t, lam, alpha_L, beta):
    A, log_term = 0.0, 0.0
    for i, ti in enumerate(t):
        if i > 0:
            A = math.exp(-beta * (ti - t[i - 1])) * (1.0 + A)      # eq. 20
        log_term += math.log(lam + alpha_L * A)
    t_k = t[-1]
    tail = sum(math.exp(-beta * (t_k - ti)) - 1.0 for ti in t)
    return log_term - lam * t_k + (alpha_L / beta) * tail

worst_laub = 0.0
for _ in range(200):
    n = random.randint(2, 40)
    T_end = random.uniform(5.0, 50.0)
    t = sorted(set(random.uniform(0, T_end) for _ in range(n)))    # distinct
    mu = random.uniform(0.05, 3.0)
    alpha = random.uniform(0.01, 0.95)
    beta = random.uniform(0.1, 4.0)
    # Laub's horizon is the last event, so evaluate ours at T = t[-1].
    ours = recursive(t, mu, alpha, beta, t[-1])[0]
    theirs = laub_eq21(t, mu, alpha * beta, beta)        # alpha_L = alpha*beta
    worst_laub = max(worst_laub, abs((-ours) - theirs) / max(1.0, abs(theirs)))

print(f"\nworst relative |-nll  -  Laub eq.21(alpha_L=alpha*beta)| : {worst_laub:.3e}")

# And confirm the substitution is not reversible: alpha_L = alpha/beta must FAIL,
# so the check above is actually discriminating.
t = [1.0, 2.5, 3.1, 4.7]
mu, alpha, beta = 0.7, 0.5, 2.3
ours = recursive(t, mu, alpha, beta, t[-1])[0]
print(f"  alpha_L = alpha*beta : {laub_eq21(t, mu, alpha * beta, beta)!r}  vs -nll {-ours!r}")
print(f"  alpha_L = alpha/beta : {laub_eq21(t, mu, alpha / beta, beta)!r}  <- wrong direction")
