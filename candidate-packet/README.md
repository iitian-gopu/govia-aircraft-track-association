# Govia — Aircraft Track Association Data Package

Version: 1.0. Fixed synthetic dataset. All identities and observations are fictional.

Use this package with the assessment brief, challenge, and clarifications. No external data, credentials, specialized hardware, or running services are needed to read it.

## Contents

| Partition | Independent scenarios | Tracklets | Unique observations |
| --- | ---: | ---: | ---: |
| train | 80 | 1,844 | 24,143 |
| validation | 20 | 455 | 6,029 |
| test | 20 | 459 | 6,059 |

Counts exclude duplicate rows and the staged demonstration. Each scenario models six distinct aircraft, typically with three or four observed fragments each. Aircraft identities are not included in inputs. Modeling labels are supplied separately.

- `data/train/scenario_*/`: training inputs and labels.
- `data/validation/scenario_*/`: validation inputs and labels.
- `data/test/scenario_*/`: final candidate test inputs and labels. Freeze your method before inspecting these labels.
- `data/staged_demo/snapshot_1/` and `snapshot_2/`: complete inputs at two receipt cutoffs, with snapshot-specific labels.
- `SOURCE_NOTES.md`: field descriptions, units, joins, and known uncertainties.
- `DATA_CONTRACT.md`: file formats, assumptions, and evaluation conventions.
- `fixtures/`: isolated valid and invalid examples for correctness tests.
- `verify_package.py`: standard-library integrity and basic structure check; not an association solution.
- `checksums.json`: SHA-256 hashes of packaged files.

## Getting started

1. Run `python3 verify_package.py` from this directory.
2. Read `DATA_CONTRACT.md` and `SOURCE_NOTES.md`.
3. Start with one training scenario, then process each scenario independently.
4. Pass only observations and supporting records into inference. Keep `labels.jsonl` and label-derived information outside the predictive input.

Each scenario contains `observations.jsonl`, `supporting.jsonl`, `labels.jsonl`, and `snapshot.json`. JSONL means one JSON object per line, encoded as UTF-8. File order is not observation order.

The manifest identifies scenario membership and file paths. Its row counts and labeled-link count are descriptive metadata, not predictive features. Do not infer aircraft identity from filenames, UUIDs, or file ordering.

A separate reviewer-only dataset exists outside this package. It is not needed to complete the exercise.
