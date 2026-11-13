"""Team Popeyes decoder for the [[18,4,4]] six-layer memory experiments.

Both ends propagate syndrome-conditioned probability to round 3. For the
per-bit logical error rate, five components suffice: the total probability
and the signed probability for each of the four observable flips.
"""

import os
from pathlib import Path

import numba as nb
import numpy as np
import stim
from numba import njit, prange

from build_tables import CircuitTables

ROOT = Path(__file__).resolve().parent
SEARCH_BEAM = 2048
RELATIVE_CUTOFF = 1e-7
RETRY_CONFIDENCE = 0.90


@njit(cache=True)
def advance(dets, layers, ptr, vals, probs, scratch, indices, weights, touched, relcut, cap):
    n = 1
    indices[0] = 0
    weights[0] = 1.
    # Track unsigned mass and four signed observable masses for each boundary syndrome.
    for layer in range(layers):
        m = 0
        best = 0.
        edges = ptr[layer]
        columns = vals[layer]
        p = probs[layer]
        for a in range(n):
            state = indices[a]
            key = state ^ dets[layer]
            cutoff = best * relcut * 0.1
            w0, w1, w2, w3, w4 = weights[a]
            for e in range(edges[key], edges[key + 1]):
                mass = w0 * p[e, 0]
                if mass < cutoff:
                    break
                s = columns[e]
                prev = scratch[s, 0]
                if prev == 0:
                    touched[m] = s
                    m += 1
                scratch[s, 0] = mass + prev
                scratch[s, 1] += w1 * p[e, 1]
                scratch[s, 2] += w2 * p[e, 2]
                scratch[s, 3] += w3 * p[e, 3]
                scratch[s, 4] += w4 * p[e, 4]
                best = max(best, mass + prev)
        if best == 0.:
            return 0
        cutoff = best * relcut
        if m > cap:  # Prune in half-octave probability bins.
            hist = np.zeros(64, np.int64)
            for i in range(m):
                mass = scratch[touched[i], 0]
                if mass >= cutoff:
                    bin_id = min(63, int(-np.log2(mass / best) * 2))
                    hist[bin_id] += 1
            cumulative = 0
            for i in range(64):
                cumulative += hist[i]
                if cumulative > cap:
                    if i > 0:
                        cutoff = max(cutoff, best * 2.0 ** (-i / 2))
                    break
        n = 0
        largest = 0
        for j in range(m):
            s = touched[j]
            mass = scratch[s, 0]
            if mass >= cutoff:
                indices[n] = s
                weights[n] = scratch[s] / best
                if mass > weights[largest, 0] * best:
                    largest = n
                n += 1
            scratch[s] = 0.
        idx = indices[largest]
        indices[largest] = indices[0]
        indices[0] = idx
        w = weights[largest].copy()
        weights[largest] = weights[0]
        weights[0] = w
    return n


@njit(cache=True)
def join(left, wl, nl, right, wr, nr, observed, local, cutoff_ratio):
    totals = np.zeros(5)
    # Independent Bernoulli faults with p <= 1/2 have maximum mass at syndrome 0.
    max_local = local[0, 0]
    best = 0.
    for a in range(nl):
        s = left[a] ^ observed
        a0, a1, a2, a3, a4 = wl[a]
        threshold = best * cutoff_ratio
        if a0 * max_local < threshold:
            break
        for b in range(nr):
            if a0 * wr[b, 0] * max_local < threshold:
                break
            key = s ^ right[b]
            mass = a0 * wr[b, 0] * local[key, 0]
            if mass < threshold:
                continue
            best = max(best, mass)
            totals[0] += mass
            totals[1] += a1 * wr[b, 1] * local[key, 1]
            totals[2] += a2 * wr[b, 2] * local[key, 2]
            totals[3] += a3 * wr[b, 3] * local[key, 3]
            totals[4] += a4 * wr[b, 4] * local[key, 4]
    return totals


@njit(cache=True)
def decode_one(
    dk, lp0, lp1, lp2, lv0, lv1, lv2, lq0, lq1, lq2, rp0, rp1, rv0, rv1,
    rq0, rq1, local, relcut, cap, scratch, li, lw, ri, rw, touched,
):
    nl = advance(
        dk, 3, (lp0, lp1, lp2), (lv0, lv1, lv2), (lq0, lq1, lq2),
        scratch, li, lw, touched, relcut, cap,
    )
    nr = advance(
        dk[::-1], 2, (rp0, rp1), (rv0, rv1), (rq0, rq1),
        scratch, ri, rw, touched, relcut, cap,
    )
    order_left = np.argsort(-lw[:nl, 0])
    order_right = np.argsort(-rw[:nr, 0])
    return join(li[order_left], lw[order_left], nl, ri[order_right], rw[order_right], nr, dk[3], local, relcut*0.1)


@njit(parallel=True)
def decode_margins(
    dets, lp0, lp1, lp2, lv0, lv1, lv2, lq0, lq1, lq2,
    rp0, rp1, rv0, rv1, rq0, rq1, local, relcut, cap, retry,
):
    out = np.zeros((len(dets), 4), np.float32)
    block = 64
    nblocks = (len(dets) + block - 1) // block
    nt = nb.get_num_threads()
    for thread in prange(nt):
        scratch = np.zeros((16384, 5))
        touched = np.empty(16384, np.int64)
        li = np.empty(16384, np.int64)
        ri = np.empty(16384, np.int64)
        lw = np.empty((16384, 5))
        rw = np.empty((16384, 5))
        for blk in range(thread, nblocks, nt):
            for i in range(blk * block, min((blk + 1) * block, len(dets))):
                dk = dets[i]
                # Revisit shots whose first-pass marginal lies close to a decision boundary.
                if retry:
                    totals = decode_one(
                        dk, lp0, lp1, lp2, lv0, lv1, lv2, lq0, lq1, lq2, rp0, rp1, rv0,
                        rv1, rq0, rq1, local, 2e-5, 256, scratch, li, lw, ri, rw, touched,
                    )
                    if totals[0] > 0:
                        confidence = np.min(np.abs(totals[1:])) / totals[0]
                        if confidence >= retry:
                            out[i] = totals[1:] / totals[0]
                            continue
                totals = decode_one(
                    dk, lp0, lp1, lp2, lv0, lv1, lv2, lq0, lq1, lq2, rp0, rp1, rv0, rv1,
                    rq0, rq1, local, relcut, cap, scratch, li, lw, ri, rw, touched,
                )
                if totals[0] > 0:
                    out[i] = totals[1:] / totals[0]
    return out


class Decoder:
    def __init__(self, circuit, cache):
        self.tables = CircuitTables(circuit, cache)
        self.predict(np.zeros((1, circuit.num_detectors), dtype=np.uint8))

    def predict(self, detector_events):
        detector_events = np.asarray(detector_events, dtype=bool)
        if detector_events.ndim != 2 or detector_events.shape[1] != len(self.tables.time):
            raise ValueError("Expected a (shots, detectors) array")
        if not len(detector_events):
            return np.empty((0, 4), dtype=np.uint8)

        packed = np.empty((len(detector_events), 6), dtype=np.int64)
        for layer, mask in enumerate(self.tables.masks):
            packed[:, layer] = detector_events[:, mask] @ self.tables.bit_weights[layer]

        # Each group of three rounds fits in a 64-bit word.
        keys = packed[:, ::3] | (packed[:, 1::3] << 14) | (packed[:, 2::3] << 28)
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        margins = decode_margins(
            np.ascontiguousarray(packed[first]), *self.tables.transitions, self.tables.middle,
            RELATIVE_CUTOFF, SEARCH_BEAM, RETRY_CONFIDENCE,
        )
        return (margins[inverse] < 0).astype(np.uint8)


_CACHE = os.environ.get("POPEYES_TABLE_CACHE")
_TABLES = []
for _basis in "XZ":
    _circuit = stim.Circuit.from_file(ROOT / f"{_basis}_circuit.stim")
    _TABLES.append((_circuit, Decoder(_circuit, _CACHE)))


def decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray:
    """Return four binary observable predictions for each detector shot."""
    for known_circuit, decoder in _TABLES:
        if circuit == known_circuit:
            return decoder.predict(detector_events)

    decoder = Decoder(circuit, _CACHE)
    _TABLES.append((circuit, decoder))
    return decoder.predict(detector_events)
