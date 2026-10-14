"""Sibling-consistency features (stage 4).

Most S1 entities have several S2/S3 records. After a first matcher pass, each candidate pair (record q, entity s)
is compared with the OTHER records confidently assigned to s ("siblings"):
  n_sib           number of confident siblings of s (excluding q)
  sib_house_eq    q's house number equals the siblings' consensus house number (1/0, -1 unknown)
  sib_house_edit  digit edit distance between q's house number and that consensus (-1 unknown)
  sib_name_max    best token_sort_ratio between q's core name and a sibling's core name (-1 none)
  sib_addr_max    best token_set_ratio between q's address and a sibling's address (-1 none)
A true record with a typo in its house number still agrees with its siblings; a decoy placed a few doors down
does not. CPU probe on v11's training set: realistic F0.5 0.98126 -> 0.98512.
"""
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

N_SIB_FEATURES = 5
SIB_NAMES = ["n_sib", "sib_house_eq", "sib_house_edit", "sib_name_max", "sib_addr_max"]


def sibling_stats(pq, ps, p1, q_house, q_name, q_addr, conf=0.5):
    """pq/ps: query / S1 index per candidate pair; p1: first-pass probability per pair.
    q_house/q_name/q_addr: per-query arrays indexed by pq. Returns float32 array (n_pairs, 5)."""
    pq, ps, p1 = np.asarray(pq), np.asarray(ps), np.asarray(p1, dtype=np.float32)
    order = np.lexsort((-p1, pq))
    first = np.r_[True, pq[order][1:] != pq[order][:-1]]
    bq, bs, bp = pq[order][first], ps[order][first], p1[order][first]
    keep = bp > conf
    sq, ss = bq[keep], bs[keep]                     # confident (query -> entity) assignments = siblings
    o = np.argsort(ss, kind="stable")
    sq, ss = sq[o], ss[o]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    ents = ss[starts]
    bounds = np.r_[starts, len(ss)]
    members, consensus = {}, {}
    for j, e in enumerate(ents):
        m = sq[bounds[j]:bounds[j + 1]]
        members[int(e)] = m
        hs = [q_house[k] for k in m if q_house[k]]
        if hs:
            vals, cnt = np.unique(hs, return_counts=True)
            consensus[int(e)] = vals[np.argmax(cnt)]
    out = np.full((len(pq), N_SIB_FEATURES), -1.0, dtype=np.float32)
    for i in range(len(pq)):
        q, s = int(pq[i]), int(ps[i])
        m = members.get(s)
        n = 0 if m is None else len(m) - int(q in m)
        out[i, 0] = n
        c = consensus.get(s, "")
        h = q_house[q]
        if c and h:
            out[i, 1] = float(h == c)
            out[i, 2] = Levenshtein.distance(h, c)
        if n:
            bn = ba = -1.0
            for k in m:
                if k == q:
                    continue
                bn = max(bn, fuzz.token_sort_ratio(q_name[q], q_name[k]))
                if q_addr[q] and q_addr[k]:
                    ba = max(ba, fuzz.token_set_ratio(q_addr[q], q_addr[k]))
            out[i, 3], out[i, 4] = bn, ba
    return out


SIB2_NAMES = SIB_NAMES + ["n_sib_s2", "n_sib_s3", "sib_house_share", "novel_tokens", "sib_p_mean"]


def sibling_stats_v2(pq, ps, p1, q_house, q_name, q_addr, q_src3, s1_name, conf=0.5):
    """sibling_stats() plus: confident siblings per source, share of siblings with q's house number,
    tokens of q's core name found in neither the S1 name nor any sibling ("novel" words: decoys add a real word
    no sibling has), and the mean first-pass probability of the siblings."""
    base = sibling_stats(pq, ps, p1, q_house, q_name, q_addr, conf)
    pq, ps, p1 = np.asarray(pq), np.asarray(ps), np.asarray(p1, dtype=np.float32)
    order = np.lexsort((-p1, pq))
    first = np.r_[True, pq[order][1:] != pq[order][:-1]]
    bq, bs, bp = pq[order][first], ps[order][first], p1[order][first]
    keep = bp > conf
    groups = {}
    for q, s, p in zip(bq[keep], bs[keep], bp[keep]):
        groups.setdefault(int(s), []).append((int(q), float(p)))
    extra = np.full((len(pq), 5), -1.0, dtype=np.float32)
    for i in range(len(pq)):
        q, s = int(pq[i]), int(ps[i])
        g = [(k, p) for k, p in groups.get(s, ()) if k != q]
        extra[i, 0] = sum(1 for k, _ in g if not q_src3[k])
        extra[i, 1] = sum(1 for k, _ in g if q_src3[k])
        if g and q_house[q]:
            hs = [q_house[k] for k, _ in g if q_house[k]]
            extra[i, 2] = sum(h == q_house[q] for h in hs) / len(hs) if hs else -1
        vocab = set(s1_name[s].split())
        for k, _ in g:
            vocab.update(q_name[k].split())
        extra[i, 3] = len(set(q_name[q].split()) - vocab)
        extra[i, 4] = np.mean([p for _, p in g]) if g else -1
    return np.hstack([base, extra])
