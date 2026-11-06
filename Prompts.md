# Govia — Technical Engineering Prompt Collection

### 1. Understand the association problem
Read the assessment and clarifications. We are matching the immediate next tracklet, not identifying aircraft. A→C must be considered wrong if an intermediate B exists. Keep associations forward in time, one-to-one and acyclic. Uncertain cases should remain unresolved rather than forcing a match.

### 2. Inspect the dataset structure
Start with the actual JSONL records. Examine scenario boundaries, tracklet counts, report frequency, observation gaps, duplicate records and missing values. Compare training and validation distributions before deciding on candidate gates or model features.

### 3. Separate observation time from availability
Use `observed_at` to reconstruct movement and `received_at` to decide when evidence became available. This distinction matters for delayed reports and staged snapshots. Ensure no report received after a snapshot cutoff influences its associations.

### 4. Validate and normalize observations
Build deterministic preprocessing. Validate UTC timestamps, coordinates and IDs. Compare duplicates using original JSON content, preserving missing-versus-null differences. Reject conflicting observation IDs and ensure changing input order does not change the result.

### 5. Investigate source-specific signals
Analyse `feed-a`, `feed-b`, `feed-c` and `telemetry-x` independently. Compare reported speed, heading and altitude against position-derived measurements. Don't assume common field names mean common units. Exclude unreliable `ref`/`tag` identifiers if they create misleading continuity evidence.

### 6. Evaluate motion estimation
Compare whole-tracklet motion fitting with endpoint and tail-window estimates. Short windows may amplify GPS noise, while whole-tracklet fitting assumes approximately stable motion. Measure prediction residuals on real training trajectories before selecting the estimator.

### 7. Design candidate generation
Use forward temporal ordering, implied speed and position prediction error to filter implausible successors. Candidate recall matters because rejected true links cannot be recovered later. Keep altitude out of the initial hard gates unless its reliability justifies excluding candidates.

### 8. Design the physical baseline
Score continuity using forward/backward position projections and altitude consistency. Normalize residuals by their expected noise and reporting gap. Avoid double-counting correlated heading and speed evidence. Keep the score interpretable and separate from calibrated probabilities.

### 9. Handle competing aircraft
Inspect nearby aircraft with similar courses. Compare the strongest alternatives rather than accepting the top candidate independently. A locally strong match may block another correct continuation. Use ambiguity margins and explicit unresolved decisions.

### 10. Prevent skipped fragments
A high-scoring A→C can still be incorrect when B should appear between them. Add a plausible-intermediate check before accepting links. Examine whether this rule suppresses valid long-gap continuations, and avoid making it overly aggressive.

### 11. Choose an association strategy
Start with deterministic greedy selection enforcing unique predecessors and successors. Compare against global assignment on difficult development cases. Only introduce a more complex optimizer if it delivers a meaningful improvement in incorrect-link prevention or recall.

### 12. Select acceptance policies
Incorrect links carry a higher cost than missed links. Compare conservative, balanced and permissive thresholds using accepted precision, recall, wrong-link count and unresolved proportion. Select operating settings on development validation and preserve them for final evaluation.

### 13. Prepare supervised examples
Reuse the shared candidate generator. Label only the immediate successor as positive; both different-aircraft pairs and same-aircraft skips are negatives. Keep candidate-pair labels separate from inference features and prevent scenario-level leakage during validation.

### 14. Train the learned model
Start with a compact gradient-boosted classifier using gap, motion residuals, speed consistency and altitude projection error. Compare a simpler model if useful. Handle missing altitude explicitly and save the feature schema, seed, parameters and model version.

### 15. Investigate model disagreement
Compare baseline and learned predictions on complete scenarios rather than pair-level classifier accuracy alone. Inspect cases where one model accepts and the other abstains. Identify whether differences originate from scoring, skip detection, competing candidates or thresholds.

### 16. Test difficult measurement conditions
Evaluate missing altitude, noisy reports, shortened tracklets and long reporting gaps with frozen settings. Specifically check short-gap noise-floor instability and long-gap classifier score collapse. Reproduce failures before changing the model, and measure regression impact.

### 17. Design the inference service
Separate training from request handling. Use FastAPI to load saved models and configurations, validate observations and return both methods' decisions. Include candidate rankings, supporting measurements, decision reasons, snapshot fingerprints and version metadata.

### 18. Build an inspectable dashboard
Examine scientific and trajectory-analysis dashboards before selecting the layout. Use Streamlit and Plotly for scenario selection, latitude/longitude trajectories, accepted and unresolved links, competing candidates and baseline/ML comparison. Keep original reports accessible and use real API responses.

### 19. Demonstrate delayed evidence
Analyse both complete staged snapshots independently with the same model and thresholds. Compare accepted, withdrawn and replaced links. Explain A→C becoming A→B→C when a delayed intermediate tracklet arrives; this changes immediate succession, not aircraft identity.

### 20. Final verification and submission
Freeze configuration before the test evaluation. Verify reproducibility, model reload, input-order independence, API errors and browser interactions. Report exact metrics, stress results and difficult cases. Keep the repository focused on runnable code, models, tests, evaluation evidence, clear architecture and quick-start instructions.

---

## Engineering Priorities

**Physical plausibility → Candidate coverage → Conservative association → Fair model comparison → Reproducible inference.**