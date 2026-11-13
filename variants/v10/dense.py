"""Unpruned per-observable likelihoods for the six-round memory circuit."""

from pathlib import Path
import hashlib
import tempfile

import numpy as np
from numba import njit, prange

from build_tables import exp_in_place, fwht, problem_info


LEFT_STAGES = ((0, 1), (1, 2), (2, 3))
RIGHT_STAGES = ((5, 4), (4, 3))
STATE_COUNT = 16384
BATCH_SIZE = 512


@njit(parallel=True, cache=True)
def walsh_rows(rows):
    for i in prange(rows.shape[0]):
        row = rows[i]
        stride = 1
        while stride < len(row):
            for start in range(0, len(row), 2 * stride):
                for j in range(start, start + stride):
                    a, b = row[j], row[j + stride]
                    row[j], row[j + stride] = a + b, a - b
            stride *= 2


def transition(H, L, p, current, following, component):
    left_bits, right_bits = len(current), len(following)
    other = np.setdiff1d(np.arange(H.shape[0]), np.r_[current, following])
    faults = np.flatnonzero(~H[other].any(axis=0) & H[current].any(axis=0))
    masks = ((1 << np.arange(left_bits)) @ H[np.ix_(current, faults)]) << right_bits
    masks |= (1 << np.arange(right_bits)) @ H[np.ix_(following, faults)]
    logs = -0.5 * np.log1p(-2 * p[faults])
    signed = logs if component == 0 else logs * (1.0 - 2.0 * L[component - 1, faults])
    spectrum = np.zeros(1 << (left_bits + right_bits))
    np.add.at(spectrum, masks, signed)
    fwht(spectrum, left_bits + right_bits)
    exp_in_place(spectrum, -logs.sum())
    fwht(spectrum, left_bits + right_bits)
    table = spectrum.astype(np.float32)
    table *= 1.0 / len(spectrum)
    return table.reshape(1 << left_bits, 1 << right_bits)


def load_transition(model, rows, stage, component, cache_dir):
    current, following = stage
    if cache_dir is None:
        return transition(*model, rows[current], rows[following], component)

    target = cache_dir / f"dense-f32-v1-{current}-{following}-{component}.npy"
    if not target.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        table = transition(*model, rows[current], rows[following], component)
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npy", delete=False) as file:
            temporary = Path(file.name)
        try:
            np.save(temporary, table)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return np.load(target, mmap_mode="r")


def propagate(packed, stages, matrices):
    values = matrices[0][packed[:, stages[0][0]]]
    for (current, _), table in zip(stages[1:], matrices[1:]):
        permutation = packed[:, current, None] ^ np.arange(values.shape[1])
        values = np.take_along_axis(values, permutation, axis=1) @ table
    return values


class DensePosterior:
    def __init__(self, circuit, tables, cache_dir):
        model = problem_info(circuit)
        rows = [np.flatnonzero(mask) for mask in tables.masks]
        if cache_dir:
            signature = hashlib.sha256(str(circuit).encode()).hexdigest()[:20]
            cache_dir = Path(cache_dir) / signature

        self.middle = np.ascontiguousarray(tables.middle.T)
        walsh_rows(self.middle)
        self.transitions = [
            tuple(load_transition(model, rows, stage, component, cache_dir)
                  for stage in (*LEFT_STAGES, *RIGHT_STAGES))
            for component in range(5)
        ]

    def signed_masses(self, packed):
        # Each component is a signed joint probability, with component zero
        # the total probability. Signs alone give the bitwise MAP decision.
        result = np.empty((len(packed), 5))
        for start in range(0, len(packed), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(packed))
            shots = packed[start:stop]
            left, left_ids = np.unique(shots[:, :3], axis=0, return_inverse=True)
            right, right_ids = np.unique(shots[:, 4:], axis=0, return_inverse=True)
            right = np.pad(right, ((0, 0), (4, 0)))
            permutation = shots[:, 3, None] ^ np.arange(STATE_COUNT)
            for component, matrices in enumerate(self.transitions):
                left_mass = propagate(left, LEFT_STAGES, matrices[:3]).astype(np.float64)
                right_mass = propagate(right, RIGHT_STAGES, matrices[3:])
                right_mass = np.take_along_axis(right_mass[right_ids], permutation, axis=1)
                right_mass = right_mass.astype(np.float64)
                walsh_rows(left_mass)
                walsh_rows(right_mass)
                result[start:stop, component] = (
                    np.sum(left_mass[left_ids] * right_mass * self.middle[component], axis=1)
                    / STATE_COUNT
                )
        return result
