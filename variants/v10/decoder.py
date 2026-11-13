"""Team Popeyes decoder for the [[18,4,4]] six-layer memory experiments.

Both ends propagate syndrome-conditioned probability to round 3. For the
per-bit logical error rate, five components suffice: the total probability
and the signed probability for each of the four observable flips. When a
v3 prediction is uncertain, recompute its likelihood with all boundary states.
"""

import math
import os
from pathlib import Path

import numba as nb
import numpy as np
import stim
from numba import njit, prange

from build_tables import CircuitTables
from dense import DensePosterior

ROOT = Path(__file__).resolve().parent
SEARCH_BEAM = 2048
RELATIVE_CUTOFF = 1e-7
RETRY_CONFIDENCE = 0.90
DENSE_THRESHOLD = 0.30


@njit(cache=True)
def advance(dets, layers, ptr, vals, probs, scratch, indices, weights, touched, hist, relcut, cap):
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
            hist[:] = 0
            for i in range(m):
                mass = scratch[touched[i], 0]
                if mass >= cutoff:
                    ratio = mass / best
                    mantissa, exponent = math.frexp(ratio)
                    if (
                        mantissa < 0.50000000000001
                        or mantissa > 0.99999999999999
                        or abs(mantissa - 0.7071067811865476) < 1e-14
                    ):
                        # Match the original binning near floating-point boundaries.
                        bin_id = min(63, int(-np.log2(ratio) * 2))
                    else:
                        bin_id = min(63, -2 * exponent + (mantissa < 0.7071067811865476))
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
                for component in range(5):
                    weights[n, component] = scratch[s, component] / best
                if mass > weights[largest, 0] * best:
                    largest = n
                n += 1
            for component in range(5):
                scratch[s, component] = 0.0
        idx = indices[largest]
        indices[largest] = indices[0]
        indices[0] = idx
        for component in range(5):
            weight = weights[largest, component]
            weights[largest, component] = weights[0, component]
            weights[0, component] = weight
    return n


@njit(cache=True)
def join(left, wl, order_left, right, wr, order_right, observed, local, cutoff_ratio):
    totals = np.zeros(5)
    # Independent Bernoulli faults with p <= 1/2 have maximum mass at syndrome 0.
    max_local = local[0, 0]
    best = 0.
    for a in order_left:
        s = left[a] ^ observed
        a0, a1, a2, a3, a4 = wl[a]
        threshold = best * cutoff_ratio
        if a0 * max_local < threshold:
            break
        for b in order_right:
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


@njit(parallel=True)
def cache_messages(segments, ptr0, ptr1, ptr2, val0, val1, val2, q0, q1, q2, relcut, cap):
    n_segments, layers = segments.shape
    capacity = min(2 * cap, 4096)
    states = np.empty((n_segments, capacity), dtype=np.int64)
    weights = np.empty((n_segments, capacity, 5))
    counts = np.zeros(n_segments, dtype=np.int64)
    n_threads = nb.get_num_threads()
    for worker in prange(n_threads):
        scratch = np.zeros((16384, 5))
        touched = np.empty(16384, np.int64)
        indices = np.empty(16384, np.int64)
        values = np.empty((16384, 5))
        hist = np.empty(64, np.int64)
        for i in range(worker, n_segments, n_threads):
            n = advance(
                segments[i], layers, (ptr0, ptr1, ptr2), (val0, val1, val2), (q0, q1, q2),
                scratch, indices, values, touched, hist, relcut, cap,
            )
            if n <= capacity:
                counts[i] = n
                order = np.argsort(-values[:n, 0])
                for j in range(n):
                    state = order[j]
                    states[i, j] = indices[state]
                    weights[i, j] = values[state]
            else:
                counts[i] = -1
    return states, weights, counts


@njit(cache=True)
def message(
    dets, cached_index, cached_states, cached_weights, cached_counts, ordered,
    layers, pointers, columns, probabilities, scratch, states, weights, touched, hist, relcut, cap,
):
    if cached_index >= 0 and cached_counts[cached_index] >= 0:
        n = cached_counts[cached_index]
        return cached_states[cached_index], cached_weights[cached_index], ordered[:n]
    n = advance(
        dets, layers, pointers, columns, probabilities,
        scratch, states, weights, touched, hist, relcut, cap,
    )
    return states, weights, np.argsort(-weights[:n, 0])


@njit(parallel=True)
def decode_margins(
    dets, left_index, right_index, left_states, left_weights, left_counts,
    right_states, right_weights, right_counts,
    lp0, lp1, lp2, lv0, lv1, lv2, lq0, lq1, lq2,
    rp0, rp1, rv0, rv1, rq0, rq1, local, relcut, cap,
):
    out = np.zeros((len(dets), 4), np.float32)
    confidence = np.zeros(len(dets))
    block = 64
    nblocks = (len(dets) + block - 1) // block
    nt = nb.get_num_threads()
    ordered = np.arange(left_states.shape[1])
    for thread in prange(nt):
        scratch = np.zeros((16384, 5))
        touched = np.empty(16384, np.int64)
        hist = np.empty(64, np.int64)
        li = np.empty(16384, np.int64)
        ri = np.empty(16384, np.int64)
        lw = np.empty((16384, 5))
        rw = np.empty((16384, 5))
        for blk in range(thread, nblocks, nt):
            for i in range(blk * block, min((blk + 1) * block, len(dets))):
                dk = dets[i]
                left, left_mass, left_order = message(
                    dk, left_index[i], left_states, left_weights, left_counts, ordered, 3,
                    (lp0, lp1, lp2), (lv0, lv1, lv2), (lq0, lq1, lq2),
                    scratch, li, lw, touched, hist, relcut, cap,
                )
                right, right_mass, right_order = message(
                    dk[::-1], right_index[i], right_states, right_weights, right_counts, ordered, 2,
                    (rp0, rp1), (rv0, rv1), (rq0, rq1),
                    scratch, ri, rw, touched, hist, relcut, cap,
                )
                totals = join(
                    left, left_mass, left_order, right, right_mass, right_order,
                    dk[3], local, relcut * 0.1,
                )
                if totals[0] > 0:
                    confidence[i] = np.min(np.abs(totals[1:])) / totals[0]
                    for bit in range(4):
                        out[i, bit] = totals[bit + 1] / totals[0]
    return out, confidence


def shared_messages(segments, transitions, relcut, cap):
    unique, inverse, counts = np.unique(segments, axis=0, return_inverse=True, return_counts=True)
    repeated = counts >= 3
    group = np.where(repeated, np.cumsum(repeated) - 1, -1)
    return group[inverse], cache_messages(unique[repeated], *transitions, relcut, cap)


def pass_margins(dets, tables, relcut, cap):
    transitions = tables.transitions
    left, right = transitions[:9], transitions[9:]
    # The cache accepts three arrays of each kind. Two layers use the first two.
    right_padded = [right[j] for j in (0, 1, 1, 2, 3, 3, 4, 5, 5)]
    li, left_cache = shared_messages(dets[:, :3], left, relcut, cap)
    ri, right_cache = shared_messages(dets[:, 5:3:-1], right_padded, relcut, cap)
    return decode_margins(
        dets, li, ri, *left_cache, *right_cache, *transitions, tables.middle, relcut, cap,
    )


class Decoder:
    def __init__(self, circuit, cache):
        self.tables = CircuitTables(circuit, cache)
        self.dense = DensePosterior(circuit, self.tables, os.environ.get("POPEYES_DENSE_CACHE"))
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
        shots = np.ascontiguousarray(packed[first])
        margins, confidence = pass_margins(shots, self.tables, 2e-5, 256)
        retry = confidence < RETRY_CONFIDENCE
        if retry.any():
            refined, _ = pass_margins(shots[retry], self.tables, RELATIVE_CUTOFF, SEARCH_BEAM)
            margins[retry] = refined
        predictions = margins < 0
        uncertain = np.flatnonzero(np.min(np.abs(margins), axis=1) < DENSE_THRESHOLD)
        if len(uncertain):
            joint = self.dense.signed_masses(shots[uncertain])
            valid = (joint[:, 0] > 0) & np.isfinite(joint).all(axis=1)
            valid &= (np.abs(joint[:, 1:]) <= joint[:, :1] * 1.0001).all(axis=1)
            predictions[uncertain[valid]] = joint[valid, 1:] < 0
        return predictions[inverse].astype(np.uint8)


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
