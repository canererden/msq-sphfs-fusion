"""m,s,q-spherical probabilistic hesitant fuzzy sets (m,s,q-SPHFS):
generalized hybrid (weighted) distance, Eq. (9)-(10) of the manuscript.

An m,s,q-SPHFS evaluated on n attributes with L hesitant elements per degree is
stored as six arrays of shape (..., n, L):
    mu, k_mu   membership values and their probabilities
    io, k_io   abstinence values and their probabilities
    nu, k_nu   non-membership values and their probabilities
Leading dimensions are free, so many objects are handled at once (broadcasting).
"""
from __future__ import annotations
import numpy as np


def _terms(a, b, m, s, q, lam):
    """Per-attribute mean |x1^p*k1 - x2^p*k2|^lam for the three degrees.
    a, b: tuples (mu, k_mu, io, k_io, nu, k_nu). Returns array (..., n, 3)."""
    out = []
    for (x1, k1, x2, k2, p) in ((a[0], a[1], b[0], b[1], m),
                                (a[2], a[3], b[2], b[3], s),
                                (a[4], a[5], b[4], b[5], q)):
        d = np.abs(np.power(x1, p) * k1 - np.power(x2, p) * k2) ** lam
        out.append(d.mean(axis=-1))
    return np.stack(out, axis=-1)


def hybrid_distance(a, b, m=3, s=3, q=3, lam=1.0, w=None):
    """Normalized generalized hybrid (weighted) distance.

    d = [ 1/(4n) * sum_z w_z * (T_mu + T_io + T_nu + max(T_mu, T_io, T_nu)) ]^(1/lam)
    w=None gives the unweighted form, Eq. (9); otherwise Eq. (10) as printed
    (the 1/(4n) factor is kept together with w, exactly as in the manuscript).
    """
    t = _terms(a, b, m, s, q, lam)                 # (..., n, 3)
    per_attr = t.sum(axis=-1) + t.max(axis=-1)     # (..., n)
    n = per_attr.shape[-1]
    if w is None:
        tot = per_attr.sum(axis=-1) / (4 * n)
    else:
        tot = (per_attr * np.asarray(w)).sum(axis=-1) / (4 * n)
    return tot ** (1.0 / lam)


def check_constraint(a, m, s, q, tol=1e-9):
    """Definition 3.1: (mu+)^m + (io+)^s + (nu+)^q <= 1 and sum of each kappa <= 1."""
    mu, k_mu, io, k_io, nu, k_nu = a
    c1 = mu.max(-1) ** m + io.max(-1) ** s + nu.max(-1) ** q <= 1 + tol
    c2 = (k_mu.sum(-1) <= 1 + tol) & (k_io.sum(-1) <= 1 + tol) & (k_nu.sum(-1) <= 1 + tol)
    return c1 & c2
