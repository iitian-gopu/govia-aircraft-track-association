# Aircraft Track Association

Offline system that decides which disconnected aircraft movement fragments ("tracklets") belong together, using a rule-based baseline and a trained model. Uncertain cases stay unresolved, and every decision is explained with its measurements.

## Quick start

Requires Python 3.11 or newer. The dataset and the trained model are in this private repository, so there's nothing to download or train.

**Windows PowerShell**

```powershell
python -m venv .venv
```
```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```
```powershell
.\.venv\Scripts\python.exe scripts\start_app.py
```

If `python` on your machine is older than 3.11, create the environment with a newer one instead, for example `py -3.13 -m venv .venv`.

**macOS or Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python scripts/start_app.py
```

Then open **http://localhost:8501**, pick a scenario and press **Run association**. Press Ctrl+C to stop everything. API documentation is at http://localhost:8000/docs.

The launcher checks that the dataset, settings and model are present and that the ports are free, and says what's wrong if not. Use `--api-port` or `--dashboard-port` if a port is taken.

## What you can explore

- **Real scenarios:** the train and validation scenarios and both staged snapshots, discovered automatically.
- **Tracks and inspector:** an interactive latitude/longitude plot with accepted links, unresolved tracklets and competing candidates, plus ranked candidates with their measurements and the reason for each decision.
- **Original records:** position reports, timestamps, altitude and supporting records for any tracklet.
- **Baseline vs learned:** decisions side by side, with a filter for disagreements.
- **Unresolved:** tracklets grouped by why no link was accepted.
- **Additional evidence:** both staged snapshots analysed separately and compared.

Ground-truth labels never reach inference or any API response, and the test partition isn't served.

## Architecture

```text
reports → validate, dedupe, order → candidate pairs (physical gates, no labels)
        → baseline score | learned score → shared selection → JSON → FastAPI → Streamlit
```

Python 3.11, NumPy, scikit-learn, FastAPI, Streamlit and Plotly, all running locally. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design, execution flow, trade-offs, source-field analysis and difficult cases.

## Baseline vs learned

Both share candidate generation and selection. A link A → B means B is the immediate next available tracklet of the same aircraft, so a link that skips a fragment is wrong.

- **Baseline:** minus the sum of squared, noise-scaled residuals for position and projected altitude. 0 is best, with no lower bound.
- **Learned:** gradient-boosted trees on gap, projection misses, altitude miss and speed difference, fitted on train only. Scores are 0 to 1 and uncalibrated.

**Test partition.** One run at settings frozen first; nothing was changed afterward. Results are in `reports/test_evaluation.json`.

| 20 scenarios, 459 tracklets | Baseline | Learned |
|---|---:|---:|
| Labeled links | 339 | 339 |
| Candidate recall | 100% | 100% |
| Correct / wrong | 338 / 0 | 338 / 0 |
| Precision, recall | 1.000, 0.997 | 1.000, 0.997 |
| Labeled but unresolved | 1 (0.29%) | 1 (0.29%) |
| Unresolved tracklets | 121 | 121 |

**Stress: altitude removed from every report at inference,** on test with the same models and settings. Baseline: 279 of 339 correct, 0 wrong, recall 0.823. Learned: 314 of 339 correct, 2 wrong, recall 0.926.

**Staged demonstration.** Both methods give the same links: 17 correct of 17 in snapshot 1 and 18 of 18 in snapshot 2, with 0 wrong. A delayed fragment arrives inside a 346.5 s gap, so `aa82aa85` → `1d397900` becomes `aa82aa85` → `691cc611` and `691cc611` → `1d397900`: 1 link replaced, 1 added, 16 unchanged. The same fragments stay connected, so this isn't a change of identity.

## Optional commands

Windows shown. On macOS or Linux, use `.venv/bin/python` and forward slashes.

```powershell
# Run the tests
.\.venv\Scripts\python.exe -m pytest -q
# Check the dataset's hashes and counts
.\.venv\Scripts\python.exe scripts\verify_data.py candidate-packet
# Reproduce the staged demonstration, writing to reports/staged/
.\.venv\Scripts\python.exe scripts\run_staged.py
```

Retrain and re-evaluate. These rewrite `models/` and `configs/`, and the pinned versions reproduce them byte for byte:

```powershell
# Train the model, then select thresholds and evaluate
.\.venv\Scripts\python.exe scripts\run_learned.py train
.\.venv\Scripts\python.exe scripts\run_learned.py select
.\.venv\Scripts\python.exe scripts\run_learned.py evaluate
# Evaluate the baseline at its saved settings
.\.venv\Scripts\python.exe scripts\evaluate_baseline.py evaluate
```

`scripts\final_test_evaluation.py run` refuses to overwrite the saved first result, by design.

## Assumptions, limitations and reproducibility

- **Signals:** only documented time, position and altitude are used. Optional attributes and supporting records are shown, not used, because they're redundant or unreliable.
- **Thresholds** were chosen on validation, ties broken on train, at an assumed cost of 5 missed links per wrong link.
- **Synthetic data:** motion is smooth and gaps are neatly bimodal, so these results aren't evidence of real-world accuracy.
- **Altitude dependence:** recall falls without it, and 300 ft of noise causes wrong links in both methods.
- **No-successor cases:** on supplied labels these tracklets have no later candidate, so 0 wrong barely tests them. With fragments withheld, the baseline makes wrong links (37 on train, 10 on validation; learned 8 and 0).
- **Learned model:** weak on long gaps. **Selection** is greedy, so an early wrong link can't be undone.
- **Data package:** `candidate-packet/` is byte for byte as supplied, for review only. `Source Notes.md` is an export of a shared Google Doc, so the supplied `verify_package.py` stops at the missing `SOURCE_NOTES.md`; `verify_data.py` makes the equivalent checks.
- **Reproducibility:** dependencies are pinned and the seed is fixed. Retraining reproduces the saved model byte for byte, and results don't change with row order or identical duplicates. Verified on Windows 11 with Python 3.11. Python 3.13 also installs and runs, with one test failing: a record nested too deep to process on 3.11 is accepted there. The macOS and Linux commands are the standard equivalents and haven't been run.

## Time and AI assistance

About 5 hours of session time, within the 6–8 hour budget, excluding the author's time outside those sessions.

