# NHL Expected Goals (xG) Model

An end-to-end machine learning system for estimating the probability that
an NHL shot results in a goal (Expected Goals, or "xG"), built as a
hands-on project in production ML engineering practices — from raw data
ingestion through model training to a deployed, automatically-updating
analytics app.

**Live app:** [link once deployed]
**Model:** XGBoost classifier, tracked and versioned via MLflow

---

## What this project does

Given a shot's location, shot type, game situation (strength, score state,
etc.), and other contextual features, the model estimates the probability
that shot becomes a goal. Those probabilities power a Streamlit app for
exploring shot data, model performance, and team-level scoring trends
across NHL seasons.

The project has three phases, all functional end-to-end:

1. **Data pipeline** — ingests play-by-play data from the NHL API into
   PostgreSQL
2. **ML model** — trains, evaluates, and registers an XGBoost model via
   MLflow
3. **Streamlit app** — serves shot-level and team-level xG analytics

Layered on top of these is a set of ML engineering practices this project
was specifically built to practice: experiment tracking, a model registry,
workflow orchestration, CI, and cloud deployment.

---

## Architecture

### Data pipeline
- Fetches game schedules and play-by-play data from the NHL API
- Parses raw play-by-play into shot-level records (location, type, strength
  state, etc.)
- Writes to PostgreSQL (`shots`, `games`, `pipeline_log` tables)
- Handles deduplication, retry logic for transient API failures, and
  re-checks for games that weren't final ("in progress") at the time of
  the nightly run

### ML model
- Feature engineering from raw shot data (`model/features.py`)
- XGBoost classifier trained via a scikit-learn `Pipeline`
  (`model/train.py`)
- Evaluation against holdout data (`model/evaluate.py`)
- Batch inference over unscored games (`model/predict.py`,
  `model/batch_predict.py`), writing results to `scored_shots` and
  `scored_games_log`
- Experiment tracking and model registry via **MLflow**; the production
  model is referenced via a registry alias
  (`models:/nhl_xg_xgboost@production`)

### App
- Built with **Streamlit** + **Plotly**
- Pages: Shot Explorer, Model Performance, Team Trends
- Reads from PostgreSQL via a connection pool
  (`psycopg2.pool.ThreadedConnectionPool`)

### Orchestration & deployment
- **Prefect** structures the pipeline and inference logic as flows/tasks,
  providing structured logging and run observability (via Prefect Cloud)
- **GitHub Actions** handles scheduling and execution:
  - `ingestion.yml` runs the nightly ingestion flow on a cron schedule
  - `inference.yml` is triggered automatically on successful completion
    of ingestion (via `workflow_run`), scoring any newly-ingested games
- **PostgreSQL** is hosted on Aiven (migrated from local Docker for
  development)
- **Streamlit Community Cloud** hosts the deployed app

This design deliberately separates scheduling/execution (GitHub Actions)
from flow structure/observability (Prefect) — see
[Design notes](#design-notes) below for why.

---

## Tech stack

| Layer | Tools |
|---|---|
| Data pipeline | Python, `requests`, PostgreSQL |
| ML | XGBoost, scikit-learn, MLflow |
| App | Streamlit, Plotly |
| Orchestration | Prefect 3.x |
| Scheduling / CI/CD | GitHub Actions |
| Database hosting | Aiven (PostgreSQL) |
| App hosting | Streamlit Community Cloud |
| Testing | pytest, `unittest.mock` |
| Linting/formatting | ruff, black |
| Security scanning | CodeQL |

---

## Known limitations & compromises

This is a solo hobby/portfolio project built under real time constraints.
The choices below were made deliberately, and are documented here rather
than left implicit — they're the kind of tradeoffs any production system
makes, just made visible.

### MLflow is not deployed as a remote service

MLflow's tracking store and model registry (`mlruns/` and `mlflow.db`)
are **committed directly to this repository** rather than hosted on a
remote tracking server or object store. This was a deliberate choice to
unblock CI/CD (GitHub Actions runners need to resolve the production
model, and they don't have access to a local filesystem) without taking
on the added infrastructure of a hosted MLflow server or a cloud
artifact store.

MLflow is genuinely used for local experiment tracking and registry 
management during model development — but the deployed pipeline is 
really just reading a static, versioned snapshot of that registry, 
not talking to a live MLflow service. Promoting a new model version 
currently requires committing the updated local store.

### Training/test split does not use out-of-fold predictions

The model was trained on the 2019–2020 through 2024–2025 seasons and
evaluated on the 2025–2026 season as a holdout test set. The test 
performance on the holdout 2025-2026 test season is therefore a valid
estimate of the model's real-world performance.

However, for shots from the **training seasons** (2019–2020 through
2024–2025), the xG values stored and displayed in the app are the
model's **in-sample predictions**, predictions on shots the model
was trained on. These values are optimistic relative to what the model
would predict on truly unseen data from those same seasons, since the
model has already learned from them directly.

The correct way to train to account for this would be using **out-of-fold (OOF) predictions**.
This involves k-fold cross-validation where each shot is only ever scored by a
model that did not see it during training, then aggregating those
per-fold predictions into a single OOF xG value per training-season shot.

---

## Future work

- [ ] Out-of-fold predictions for training-season shots, for honest
      historical xG values
- [ ] Automated model retraining flow with performance-gated promotion
- [ ] Metric regression gates in CI
- [ ] Data validation at pipeline boundary (e.g. pandera)
- [ ] Host MLflow tracking/registry remotely (hosted tracking server or
      cloud object store) rather than committing `mlruns/`/`mlflow.db`