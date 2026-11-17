"""How the routing op's device tests judge a launch against the float64 oracle, in one place so a host test can prove
that the judgement itself goes red.

Per token: the indices as a SET (the slot order is not the contract), the weights paired by expert id.
- Equal sets: the weights are compared with the oracle's, relative to the token's largest weight.
- Different sets: allowed ONLY on a near-tie, that is a reference gap ABOVE 0 and at most the band (two choice bounds;
  the gap between the k-th and (k+1)-th choice values, or half the gap between the last kept and first dropped group
  score). An EXACT tie (gap 0: the lanes of a dropped group all hold the fill) is not a near-tie: the op's rule (lower
  id first) fixes the set and the reference set is required. On a near-tie the weights are still compared, on the ids
  the device chose: the reference weights of THAT set are computed from the float64 scores (renormalised over the
  device's set), so a wrong weight cannot hide behind a different set.
The near-tie reading uses the reference alone, so a wrong kernel cannot widen it."""
import math

import numpy as np


def gap_of(v, n):
    """The gap between the n-th and (n+1)-th largest values of a row (inf when there is no (n+1)-th)."""
    srt = np.sort(v)[::-1]
    return math.inf if n >= len(srt) else float(srt[n - 1] - srt[n])


def scores_of(x, w, a):
    """The float64 scores (before the bias) of one token row: sigmoid or softmax of the router product."""
    z = w @ x
    if a["score_fn"] == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z - z.max())
    return e / e.sum()


def weights_of_set(ids, score, a):
    """The reference weights a set of expert ids gets: its scores, renormalised when asked, times the scale."""
    s = score[np.asarray(ids)]
    if a["renorm_eps"] is not None:
        s = s / (s.sum() + a["renorm_eps"])
    return s * a["scale"]


def compare(idx, wt, ref, a, name, xr, wr, weights_bound, choice_bound):
    """The worst weight deviation (relative to the largest weight of the token) and the near-tie tokens (token, gap).
    Raises AssertionError on a wrong set, a set with a repeated or out-of-range id, or an exact tie not resolved as the
    reference does. The caller asserts the weight bound."""
    ri, rw, rc, rg = ref
    band = 2 * choice_bound
    worst, ties = 0.0, []
    for t in range(idx.shape[0]):
        assert len(set(idx[t].tolist())) == a["top_k"] and idx[t].min() >= 0 and idx[t].max() < rc.shape[1], \
            f"{name} token {t}: indices {idx[t].tolist()} are not {a['top_k']} distinct experts"
        if set(idx[t].tolist()) == set(ri[t].tolist()):
            mine, theirs = wt[t][np.argsort(idx[t])], rw[t][np.argsort(ri[t])]
        else:
            gc = gap_of(rc[t], a["top_k"])
            gg = gap_of(rg[t], a["topk_group"]) / 2 if a["n_group"] > 1 else math.inf
            gap = min(gc, gg)
            assert 0 < gap <= band, (f"{name} token {t}: set {sorted(idx[t].tolist())} against "
                                     f"{sorted(ri[t].tolist())}, reference gap {gap} is "
                                     f"{'an exact tie: lower id first fixes the set' if gap == 0 else 'above the band'}")
            ties.append((t, gap))
            ids = np.sort(idx[t])
            mine, theirs = wt[t][np.argsort(idx[t])], weights_of_set(ids, scores_of(xr[t], wr, a), a)
        worst = max(worst, float(np.abs(mine - theirs).max() / np.abs(theirs).max()))
    return worst, ties


def plain_route(x, w, b, a):
    """The routing written plainly, token by token in float64, from the op's definition alone (no numpy sort, no
    oracle code): (indices [T, k], weights [T, k], choice values [T, E] after the fill, group scores [T, n_group]),
    the tuple `compare` reads. A rank is a list sorted on (-value, id): the larger value first, the lower id on a
    tie. `x` [T, H], `w` [E, H], `b` [E] or None, at the values the kernel reads."""
    x, w = np.asarray(x, np.float64), np.asarray(w, np.float64)
    T, E, G = x.shape[0], w.shape[0], a["n_group"]
    per = E // G
    out_i, out_w, out_c, out_g = [], [], [], []
    for t in range(T):
        logit = [float(np.dot(w[e], x[t])) for e in range(E)]
        if a["score_fn"] == "sigmoid":
            s = [1.0 / (1.0 + math.exp(-v)) for v in logit]
        else:
            top = max(logit)
            ex = [math.exp(v - top) for v in logit]
            s = [v / sum(ex) for v in ex]
        c = [s[e] + (0.0 if b is None else float(b[e])) for e in range(E)]
        gscore = []
        for g in range(G):
            members = sorted(c[g * per:(g + 1) * per], reverse=True)
            gscore.append(members[0] + members[1] if a["group_score"] == "top2_sum" and G > 1 else members[0])
        kept = sorted(range(G), key=lambda g: (-gscore[g], g))[:a["topk_group"]]
        c = [c[e] if e // per in kept else a["fill"] for e in range(E)]
        ids = sorted(range(E), key=lambda e: (-c[e], e))[:a["top_k"]]
        wt = [s[e] for e in ids]
        if a["renorm_eps"] is not None:
            total = sum(wt) + a["renorm_eps"]
            wt = [v / total for v in wt]
        out_i.append(ids)
        out_w.append([v * a["scale"] for v in wt])
        out_c.append(c)
        out_g.append(gscore)
    return (np.array(out_i, np.int64), np.array(out_w), np.array(out_c), np.array(out_g))
