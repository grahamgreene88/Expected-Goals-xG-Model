# NHL Expected Goals (xG) Model

An end-to-end machine learning system for estimating the probability that
an NHL shot results in a goal (Expected Goals, or "xG"), built as a
hands-on project in production ML engineering practices — from raw data
ingestion through model training to a deployed, automatically-updating
analytics app.

**Live app:** [https://xg-web-app.onrender.com]
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

The system has two loops: a manual local loop for training and promoting
models, and an automated GitHub Actions loop that ingests new games nightly
and scores them with the committed production model.

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
- Experiment tracking and model registry via **MLflow**, used locally
  during development. The model version holding the `production` alias
  is exported to `production_model/` and committed; batch inference
  loads it from that repo-relative path

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
- **PostgreSQL** is hosted on Aiven in production, Docker Compose locally for development
- **Render** hosts the deployed app

This design deliberately separates scheduling/execution (GitHub Actions)
from flow structure/observability (Prefect).

---

## Tech stack

| Layer | Tools |
|---|---|
| Data pipeline | Python, `requests`, PostgreSQL |
| ML | XGBoost, scikit-learn, MLflow |
| App | Streamlit, Plotly |
| Orchestration, scheduling & CI/CD | GitHub Actions |
| Containerization | Docker (app image), Docker Compose (local PostgreSQL) |
| Database hosting | Aiven (PostgreSQL) |
| App hosting | Render (Docker deploy) |
| Testing | pytest, unittest |
| Environment & dependencies | uv |
| Linting/formatting | ruff, black |
| Security scanning | CodeQL |

---

## Known limitations & compromises

This is a solo hobby/portfolio project built under real time constraints.
The choices below were made deliberately, and are documented here rather
than left implicit — they're the kind of tradeoffs any production system
makes, just made visible.

### MLflow is not deployed as a remote service

MLflow is mainly used for local experiment tracking and registry 
management during model development.

MLflow's tracking store and model registry (`mlruns/` and `mlflow.db`)
run locally rather than on a remote tracking server or object store. A
locally written registry records artifact locations as absolute paths,
so it cannot be resolved on another machine such as a GitHub Actions
runner.

Batch inference therefore does not read the registry. Promoting a model
is a manual, local procedure:

1. Assign the `production` alias to the model version in the local registry.
2. Run `python -m model.export_model`, which copies that version's files
   to `production_model/model/` and writes its version label and run ID
   to `production_model/model_version.json`.
3. Commit `production_model/`.

`mlruns/` and `mlflow.db` remain committed as an off-machine backup of
experiment history; nothing in the deployed pipeline reads them.

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

The correct way to produce historical xG values would be using **out-of-fold (OOF) predictions**:
k-fold cross-validation where each training-season shot is scored only by a model that never saw it.
Folds would need to be grouped by season rather than split randomly by shot. Shots in the same game share context (goalie, score state, rebound sequences), so a random split would leak information across folds and inflate the OOF values.

---

## Future work

- [ ] Out-of-fold predictions for training-season shots, for honest
      historical xG values
- [ ] Automated model retraining flow with performance-gated promotion
- [ ] Metric regression gates in CI
- [ ] Data validation at pipeline boundary (e.g. pandera)
- [ ] Host MLflow tracking/registry remotely (hosted tracking server or
      cloud object store) rather than committing `mlruns/`/`mlflow.db`
