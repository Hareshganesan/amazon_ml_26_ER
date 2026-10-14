"""F0.5-optimal per-entity decisions (instead of one global threshold).

Macro F0.5 is an average over S1 entities, so the right decision for a record depends on the other records the same
entity has: e.g. a singleton scores 1.0 only if NOTHING is predicted, while an entity with four confident records
barely notices one extra borderline record. Given calibrated probabilities q_i that each record (whose best S1 is
entity e) truly belongs to e, and treating them as independent, the prediction maximising E[F0.5] is a top-k set by
q. For every entity we evaluate E[F0.5 | top-k] exactly (Poisson-binomial counts of true records inside / outside
the chosen set, plus an expected number `miss` of true records that are not candidates at all) and keep the best k.

    F0.5 = 1.25 A / (1.25 A + 0.25 (B + M) + (k - A)),  A = true in top-k, B = true outside, M = missed elsewhere;
    k = 0 scores 1 only if A = B = M = 0 (singleton), else 0.

calibrate(p, correct) -> isotonic map; fopt_keep(entity, q, miss) -> boolean keep mask per record.
"""
import numpy as np


def calibrate(p, correct):
    """Fit an isotonic regression mapping model probability p to the observed correctness rate."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(p, correct)
    return iso


def _pb(Q):
    """Poisson-binomial pmf of the number of successes for each row of Q (E x m) -> (E x m+1)."""
    E, m = Q.shape
    P = np.zeros((E, m + 1))
    P[:, 0] = 1.0
    for j in range(m):
        q = Q[:, j:j + 1]
        P[:, 1:j + 2] = P[:, 1:j + 2] * (1 - q) + P[:, 0:j + 1] * q
        P[:, 0:1] = P[:, 0:1] * (1 - q)
    return P


def _expected_f(Q, k, miss_pmf):
    """E[F0.5] of predicting the top-k records (columns 0..k-1 of the descending-sorted Q)."""
    E, n = Q.shape
    PA = _pb(Q[:, :k])            # true among chosen
    PB = _pb(Q[:, k:])            # true among the rest of the candidates
    out = np.zeros(E)
    for a in range(k + 1):
        for b in range(n - k + 1):
            w = PA[:, a] * PB[:, b]
            for m, pm in enumerate(miss_pmf):
                if k == 0:
                    f = 1.0 if (a + b + m) == 0 else 0.0
                elif a == 0:
                    f = 0.0
                else:
                    f = 1.25 * a / (1.25 * a + 0.25 * (b + m) + (k - a))
                if f:
                    out += w * pm * f
    return out


def fopt_keep(entity, q, miss=0.03, max_n=12, fallback_t=0.6):
    """entity: S1 id/position of each record's best candidate; q: calibrated probability that it is right.
    Returns a boolean keep mask. Entities with more than max_n records fall back to q > fallback_t."""
    entity, q = np.asarray(entity), np.asarray(q, dtype=np.float64)
    keep = np.zeros(len(q), bool)
    miss_pmf = [np.exp(-miss), 1 - np.exp(-miss)]  # P(no / some true record lies outside the candidates)
    order = np.lexsort((-q, entity))
    ent_sorted = entity[order]
    starts = np.r_[0, np.flatnonzero(ent_sorted[1:] != ent_sorted[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    for n in np.unique(sizes):
        grp = starts[sizes == n]
        idx = grp[:, None] + np.arange(n)[None, :]          # positions in `order`
        if n > max_n:
            keep[order[idx.ravel()]] = q[order[idx.ravel()]] > fallback_t
            continue
        Q = q[order[idx]]
        ef = np.stack([_expected_f(Q, k, miss_pmf) for k in range(n + 1)], axis=1)
        best_k = ef.argmax(axis=1)
        sel = np.arange(n)[None, :] < best_k[:, None]
        keep[order[idx[sel]]] = True
    return keep
