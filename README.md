# Team Popeyes | Quantum Error Correction Decoder Hackathon

**Kaggle team name: Popeyes**

## Competition hand-in

**Kaggle submission:** `submission_v3.csv` (public LER **0.03361**).

The hand-in entry point is [`decoder.py`](decoder.py) at the repository root. It uses the prefix-cached v3 decoder and reproduces all 200,000 rows of the selected CSV byte for byte. The original v3 implementation is also available in [`variants/v3/`](variants/v3/).

The root decoder was verified in a fresh Python 3.13 environment using only the pinned [requirements](requirements.txt). It builds its lookup tables at import from the two bundled Stim circuits using [`build_tables.py`](build_tables.py). The decoder runs without network access or an installation of the organizers' `hackathon_2026` package.

## Decoder method

We decode the four logical observables of the `[[18,4,4]]` memory circuit. The decoder builds transition probabilities from the Stim detector error model and propagates them from both ends of the circuit. It estimates each logical bit from its posterior marginal, matching the competition's per-bit error metric.

## Archived decoder versions

| Submission | Source | Notes |
| --- | --- | --- |
| v3 | [`decoder.py`](decoder.py) | Prefix-cached hand-in decoder. |
| v3 | [`variants/v3/decoder.py`](variants/v3/decoder.py) | Original implementation behind v3. |
| v8 | [`variants/v8/decoder.py`](variants/v8/decoder.py) | Uses the next detector round to guide search on uncertain shots. |
| v9 | [`variants/v9/decoder.py`](variants/v9/decoder.py) | Uses the likelihood from the opposite half of the circuit to guide search. |
| v10 | [`variants/v10/decoder.py`](variants/v10/decoder.py) | Recomputes uncertain shots using all boundary states. |

Each directory exports `decode(circuit: stim.Circuit, detector_events: numpy.ndarray) -> numpy.ndarray`. Inputs have shape `(shots, 70)` and outputs have shape `(shots, 4)`, ordered from `L0` through `L3`. The bundled circuits and table-building code reconstruct the tables locally at import. For repeat runs, set `POPEYES_TABLE_CACHE` to a writable directory. V10 also accepts `POPEYES_DENSE_CACHE` for its dense matrices, which take about 30 GB of storage.

## Reproduce the v3 submission

From the repository root, use a Python 3.13 environment and install the pinned requirements. Pass the path to the competition's `test.csv`. The script also reports logical error rate when its input has all four observable labels.

```bash
python -m pip install -r requirements.txt
export POPEYES_TABLE_CACHE="$(mktemp -d)"
python run_local.py /path/to/test.csv --out reproduced_v3.csv --threads 12 --runs 5
sha256sum reproduced_v3.csv
```

On macOS, use `shasum -a 256 reproduced_v3.csv` for the checksum. The runner reports the median time across five runs and checks that predictions agree across runs. The timing excludes imports and table construction. For a local comparison, use `--variant v3` for the original implementation or `--variant v8`, `--variant v9`, or `--variant v10` for an archived variant.

The expected SHA-256 values below are from the supplied 200,000-shot test set.

| Submission | SHA-256 |
| --- | --- |
| v3 | `3e6d46af69e3b031cc9c67f776fd2793805bcee57b32318af235efafb348f6ba` |
| v8 | `f4423bf6095e233bdabc14f8d2a3a06dd449cabdae3934f3bbb7cf159888be1b` |
| v9 | `ec69b683ca729ffb694a83a0226fd0103ca30db28a4c2ad453cbf2c48f677cd4` |
| v10 | `7bdf0f545d523b5583b817327d67a5485fb0c6c8dd47c252b22d8645b49d2d69` |

The organizers supplied the circuits and the basis for the CSV helpers. We used Codex for implementation and optimization assistance. See the [upstream license](UPSTREAM-LICENSE) and the [organizers' repository](https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026).
