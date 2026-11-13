"""Decode and optionally score a competition CSV."""

import argparse
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("v3-fast", "v3", "v8", "v9", "v10"), default="v3-fast")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or args.runs < 1:
        parser.error("--threads and --runs must be positive")
    if args.input.resolve() == args.out.resolve():
        parser.error("--out must be different from the input CSV")

    for setting in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[setting] = str(args.threads)

    import numpy as np
    import stim
    from csv_io import load_kaggle_csv, save_submission

    try:
        data = load_kaggle_csv(args.input)
    except ValueError as error:
        parser.error(str(error))
    if not np.isin(data["basis"], ("X", "Z")).all():
        parser.error("input CSV must use X or Z as the basis")

    directory = ROOT if args.variant == "v3-fast" else ROOT / "variants" / args.variant
    sys.path.insert(0, str(directory))
    from decoder import decode

    experiments = []
    for basis in ("X", "Z"):
        rows = np.flatnonzero(data["basis"] == basis)
        if len(rows):
            circuit = stim.Circuit.from_file(directory / f"{basis}_circuit.stim")
            experiments.append((rows, circuit, data["detector_events"][rows]))

    times = []
    first = None
    for _ in range(args.runs):
        predictions = np.empty((len(data["id"]), 4), dtype=np.uint8)
        start = time.perf_counter()
        outputs = [decode(circuit, events) for _, circuit, events in experiments]
        times.append(time.perf_counter() - start)
        for (rows, _, _), output in zip(experiments, outputs):
            if output.shape != (len(rows), 4) or not np.isin(output, (0, 1)).all():
                raise ValueError("decoder must return four binary predictions per shot")
            predictions[rows] = output
        if first is None:
            first = predictions
        elif not np.array_equal(first, predictions):
            raise RuntimeError("decoder predictions differed across runs")

    save_submission(args.out, data["id"], first)
    print(f"Wrote {len(first):,} predictions to {args.out}")
    print(f"Median decode time over {args.runs} run(s) was {statistics.median(times):.2f} s")
    if "observable_flips" in data:
        print(f"Logical error rate was {np.mean(first != data['observable_flips']):.8f}")


if __name__ == "__main__":
    main()
