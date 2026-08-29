"""Cardinality-constrained meet-in-the-middle subset sum (Horowitz-Sahni, 1974).

Why this exists: naive enumeration of subsets up to size k over a candidate pool
of n is C(n, <=k), which is fine at k=3 and unusable at k=8. Measurement showed
the k=3 bound was not a minor limitation — once the generator produced realistic
end-of-day consolidations of 3 to 8 payouts, settlement match rate collapsed from
99.42% to 37.74%. Sixty-one points of recall were sitting behind that bound.

MITM splits the pool in half, enumerates subset sums of each half under the
cardinality cap, sorts one side and binary-searches the complement. Time goes
from C(n, <=k) to roughly 2 * C(n/2, <=k) plus a log factor.

The critical addition for this system is **ambiguity detection**. The previous
enumerator returned the first subset that hit the target, which is a latent false
-match path: if two different subsets both sum to a bank credit, picking either
one is a coin flip. This returns up to `solution_cap` DISTINCT solutions so the
caller can refuse to assert when the answer is not unique. Raising k without that
gate would have traded the zero-false-match invariant for recall.

Everything is integer paise. No floats.
"""

from __future__ import annotations

import bisect
from itertools import combinations

# Hard guards. Beyond these the caller falls back and reports an exception
# rather than burning unbounded time inside a reconciliation run.
MAX_POOL = 40
MAX_K = 10


def _half_sums(indices: list[int], amounts: list[int], kmax: int,
               witness_cap: int = 3) -> dict[int, list[tuple[int, ...]]]:
    """All subset sums of one half, up to cardinality kmax.

    Maps sum -> up to `witness_cap` distinct index-tuples producing it. Keeping
    several witnesses per sum is what lets the caller notice that a target is
    reachable more than one way.
    """
    out: dict[int, list[tuple[int, ...]]] = {0: [()]}
    for k in range(1, min(kmax, len(indices)) + 1):
        for comb in combinations(indices, k):
            s = sum(amounts[i] for i in comb)
            bucket = out.setdefault(s, [])
            if len(bucket) < witness_cap:
                bucket.append(comb)
    return out


def subset_sum(amounts: list[int], target: int, max_k: int = 8, tol: int = 0,
               min_k: int = 2, solution_cap: int = 2) -> list[tuple[int, ...]]:
    """Find up to `solution_cap` DISTINCT index subsets summing to target +/- tol.

    Returns index tuples into `amounts`, each of size in [min_k, max_k].
    A return of length > 1 means the answer is ambiguous and the caller must not
    assert a match. A return of length 0 means no subset explains the target.
    """
    n = len(amounts)
    if n == 0 or n > MAX_POOL:
        return []
    max_k = min(max_k, MAX_K, n)

    idx = list(range(n))
    mid = n // 2
    left, right = idx[:mid], idx[mid:]

    L = _half_sums(left, amounts, max_k)
    R = _half_sums(right, amounts, max_k)
    r_sums = sorted(R)

    found: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    for s_left, w_left in L.items():
        need = target - s_left
        lo = bisect.bisect_left(r_sums, need - tol)
        hi = bisect.bisect_right(r_sums, need + tol)
        for j in range(lo, hi):
            for wl in w_left:
                for wr in R[r_sums[j]]:
                    size = len(wl) + len(wr)
                    if size < min_k or size > max_k:
                        continue
                    key = tuple(sorted(wl + wr))
                    if key in seen:
                        continue
                    seen.add(key)
                    found.append(key)
                    if len(found) >= solution_cap:
                        return found
    return found


def is_unique(amounts: list[int], target: int, max_k: int = 8, tol: int = 0,
              min_k: int = 2) -> tuple[bool, tuple[int, ...] | None]:
    """Convenience wrapper: (unique, the_solution).

    `unique` is True only when exactly one distinct subset explains the target.
    """
    sols = subset_sum(amounts, target, max_k=max_k, tol=tol, min_k=min_k,
                      solution_cap=2)
    if len(sols) == 1:
        return True, sols[0]
    return False, (sols[0] if sols else None)
