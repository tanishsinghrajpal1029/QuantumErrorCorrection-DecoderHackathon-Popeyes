"""Build transition tables directly from the circuit's detector error model."""

import hashlib
from pathlib import Path
import tempfile

import numpy as np
from numba import njit, prange

TABLE_CAP = 4096
TABLE_CUTOFF = 1e-8
TABLE_VERSION = "1"
SIGN = (1 - 2 * ((np.arange(16)[:, None] >> np.arange(4)) & 1)).astype(np.float64)


def problem_info(circuit):
    # Merge identical independent errors by their odd-parity probability.
    probabilities = {}
    for instruction in circuit.detector_error_model().flattened():
        if instruction.type != "error":
            continue
        detectors = set()
        observables = set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors ^= {target.val}
            elif target.is_logical_observable_id():
                observables ^= {target.val}
        signature = (frozenset(detectors), frozenset(observables))
        p = instruction.args_copy()[0]
        q = probabilities.get(signature, 0.0)
        probabilities[signature] = q * (1 - p) + p * (1 - q)

    H = np.zeros((circuit.num_detectors, len(probabilities)), dtype=np.uint8)
    L = np.zeros((circuit.num_observables, len(probabilities)), dtype=np.uint8)
    for j, (detectors, observables) in enumerate(probabilities):
        H[list(detectors), j] = 1
        L[list(observables), j] = 1
    return H, L, np.array(list(probabilities.values()))


@njit(parallel=True, cache=True)
def fwht(x, bits):
    tile_bits = min(bits, 11)
    tile = 1 << tile_bits
    for block in prange(len(x) // tile):
        start = block * tile
        for bit in range(tile_bits):
            stride = 1 << bit
            for lo in range(start, start + tile, 2 * stride):
                for j in range(lo, lo + stride):
                    a, b = x[j], x[j + stride]
                    x[j], x[j + stride] = a + b, a - b
    for bit in range(tile_bits, bits):
        stride = 1 << bit
        for block in prange(len(x) // (2 * tile)):
            lo = (block // (stride // tile)) * 2 * stride + (block % (stride // tile)) * tile
            for j in range(lo, lo + tile):
                a, b = x[j], x[j + stride]
                x[j], x[j + stride] = a + b, a - b


@njit(parallel=True, cache=True)
def exp_in_place(x, offset):
    for i in prange(len(x)):
        x[i] = np.exp(x[i] + offset)


@njit(parallel=True, cache=True)
def pick(data, nrows, ncols, cap, rel):
    counts = np.zeros(nrows, dtype=np.int64)
    cols = np.empty((nrows, cap), dtype=np.uint16)
    for r in prange(nrows):
        row = data[r * ncols : (r + 1) * ncols]
        cutoff = max(1e-18, rel * row.max())
        candidates = np.empty(ncols, dtype=np.int64)
        n = 0
        for j in range(ncols):
            if row[j] >= cutoff:
                candidates[n] = j
                n += 1
        k = min(n, cap)
        order = np.argsort(-row[candidates[:n]])
        for j in range(k):
            cols[r, j] = candidates[order[j]]
        counts[r] = k
    return counts, cols


@njit(parallel=True, cache=True)
def gather(data, ptr, cols, weights, ncols, component):
    scale = 1.0 / len(data)
    for r in prange(len(ptr) - 1):
        for i in range(ptr[r], ptr[r + 1]):
            weights[i, component] = data[r * ncols + cols[i]] * scale


def distribution(H, L, p, curr, following, cap, rel):
    # A Walsh transform turns each XOR convolution into multiplication.
    # Transforming the log-multipliers first builds the complete spectrum
    # using only five arrays in sequence, one for each required marginal.
    nl, nr = len(curr), len(following)
    nrows, ncols = 1 << nl, 1 << nr
    other = np.setdiff1d(np.arange(H.shape[0]), np.r_[curr, following])
    cols = np.flatnonzero(~H[other].any(axis=0) & H[curr].any(axis=0))
    mask = ((1 << np.arange(nl)) @ H[np.ix_(curr, cols)]) << nr
    mask |= (1 << np.arange(nr)) @ H[np.ix_(following, cols)]
    logs = np.log1p(-2 * p[cols]) * -0.5
    offset = -logs.sum()
    ptr = dest = weights = None
    for q in range(5):
        x = np.zeros(nrows * ncols)
        signed = logs if q == 0 else logs * (1.0 - 2.0 * L[q - 1, cols])
        np.add.at(x, mask, signed)
        fwht(x, nl + nr)
        exp_in_place(x, offset)
        fwht(x, nl + nr)
        if q == 0:
            counts, positions = pick(x, nrows, ncols, min(cap, ncols), rel)
            ptr = np.r_[0, np.cumsum(counts)]
            dest = positions[np.arange(positions.shape[1])[None,:] < counts[:, None]]
            weights = np.empty((ptr[-1], 5), dtype=np.float32)
        gather(x, ptr, dest, weights, ncols, q)
        del x
    for q in range(1,5):
        np.clip(weights[:, q], -weights[:, 0], weights[:, 0], out=weights[:, q])
    return ptr, dest, weights


@njit(cache=True)
def _dense_apply(distribution, masks, probabilities):
    for mask, p in zip(masks, probabilities):
        for i in range(len(distribution)):
            j = i ^ mask
            if i < j:
                a, b = distribution[i], distribution[j]
                distribution[i] = (1 - p) * a + p * b
                distribution[j] = (1 - p) * b + p * a


def middle_distribution(H, L, p, time):
    rows = np.flatnonzero(time == 3)
    cols = np.flatnonzero(~H[time != 3].any(axis=0))
    synd = (1 << np.arange(len(rows))) @ H[np.ix_(rows, cols)]
    obs = (1 << np.arange(4)) @ L[:, cols]
    q = np.zeros((1 << len(rows)) * 16)
    q[0] = 1.0
    _dense_apply(q, (synd << 4) | obs, p[cols])
    q = q.reshape(-1, 16)
    return np.column_stack((q.sum(axis=1), q @ SIGN))


class CircuitTables:
    def __init__(self, circuit, cache):
        coords = circuit.get_detector_coordinates()
        self.time = np.array([int(coords[i][-1]) for i in range(circuit.num_detectors)])
        expected = [7, 14, 14, 14, 14, 7]
        if circuit.num_observables != 4 or np.bincount(self.time).tolist() != expected:
            raise ValueError("Expected the six-layer [[18,4,4]] memory circuit")

        self.masks = [self.time == layer for layer in range(6)]
        self.bit_weights = [1 << np.arange(width) for width in expected]
        self.middle, self.transitions = self._load(circuit, Path(cache) if cache else None)

    def _load(self, circuit, cache):
        H, L, p = problem_info(circuit)
        for detector_mask in H.T:
            layers = self.time[detector_mask.astype(bool)]
            if len(layers) and np.ptp(layers) > 1:
                raise ValueError("An error mechanism spans more than two adjacent layers")
        if np.any(p >= 0.5):
            raise ValueError("Expected error probabilities below 1/2")

        middle = middle_distribution(H, L, p, self.time)
        signature = hashlib.sha256(str(circuit).encode()).hexdigest()[:20]
        transitions = []
        for direction in (((0, 1), (1, 2), (2, 3)), ((5, 4), (4, 3))):
            tables = []
            for current, following in direction:
                path = cache / f"{signature}-{current}-{following}-v{TABLE_VERSION}.npz" if cache else None
                if path and path.exists():
                    with np.load(path) as saved:
                        table = tuple(saved[str(k)] for k in range(3))
                else:
                    table = distribution(
                        H, L, p, np.flatnonzero(self.masks[current]),
                        np.flatnonzero(self.masks[following]), TABLE_CAP, TABLE_CUTOFF,
                    )
                    if path:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as output:
                            temporary_path = Path(output.name)
                        try:
                            np.savez(temporary_path, **{str(k): array for k, array in enumerate(table)})
                            temporary_path.replace(path)
                        finally:
                            temporary_path.unlink(missing_ok=True)
                tables.append(table)
            transitions.extend(array for group in zip(*tables) for array in group)
        return middle, transitions
