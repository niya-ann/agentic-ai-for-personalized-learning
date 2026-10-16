# Agentic AI for Personalized Learning — Level 2 Prototype

This is a working implementation of the Level 2 prototype from your feasibility
document: 4 agents + orchestrator, a NetworkX Knowledge Graph, a neuro-symbolic
verification layer, structured XAI explanations, a full HITL dashboard, and a
real Flower FedAvg experiment — built on your **OULAD dataset** (used for the
risk-classifier / Federated Learning part) plus a synthetic learner generator
(used for the live Learner/Educator demo, since OULAD has no concept-level
curriculum data).

---

## 0. Where to build this — VS Code, not Colab/Kaggle

Build and run this **locally in VS Code**, not in a Colab or Kaggle notebook.
Reasons, matching Section 6 of your feasibility doc:

- Nothing here needs a GPU. SQLite, NetworkX, scikit-learn, and Flower's FedAvg
  simulation all run on CPU in seconds. This repo's FL experiment (15 rounds,
  4 institutions, ~26k real OULAD rows) finished in a few seconds on an ordinary
  CPU during testing.
- This is a **multi-process local web app** (a FastAPI backend + a Streamlit
  frontend + a SQLite file, running at the same time, talking to each other over
  `localhost`). Notebooks (Colab/Kaggle) are built around one cell executing at a
  time in one process — they fight this architecture. A normal terminal + VS Code
  is the right tool.
- Everything is free, offline, and zero-cost on an 8GB RAM laptop (Section 22/23).

Only reach for Colab/Kaggle later if you add a 7B local LLM (Phase 9, optional,
not required for Level 2) and want free GPU minutes for it.

---

## 1. One-time setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Put the OULAD CSVs here (already done if you extracted the zip as below):
#    data/oulad/assessments.csv
#    data/oulad/courses.csv
#    data/oulad/studentAssessment.csv
#    data/oulad/studentInfo.csv
#    data/oulad/studentRegistration.csv
#    data/oulad/studentVle.csv
#    data/oulad/vle.csv
```

VS Code: open this folder (`File > Open Folder`), install the Python extension
if prompted, and select the `venv` interpreter (bottom-right corner or
Ctrl+Shift+P → "Python: Select Interpreter").

---

## 2. Build the data (run once, in order)

```bash
# a) Build the Knowledge Graph (auto-runs on first import too, but do it explicitly once)
python -m app.kg

# b) Turn OULAD into risk-model features, partitioned by institution (code_module)
python -m app.data.oulad_prep

# c) Populate the app with synthetic demo learners (for the Learner/Educator UI)
python -m app.data.generate_synthetic --n 60
```

You should now have these files in `db_files/`:
`app.db`, `kg.json`, `oulad_features.csv`

---

## 3. Train the risk model + run the Federated Learning experiment

```bash
# Centralized baseline (all institutions' data pooled — the comparison point)
python -m app.risk_model.train_baseline

# The actual FL experiment: 4 simulated institutions (OULAD's BBB/FFF/DDD/CCC
# course modules), real Flower NumPyClient objects, real FedAvg aggregation
python -m app.fl_experiment.run_fl_experiment
```

This writes `db_files/fl_results.json`, `db_files/fl_accuracy_chart.png`, and
`db_files/baseline_results.json` — this chart and these numbers are your
Federated Learning evidence for the dissertation (Section 17).

**Note on how the FL experiment is driven:** `run_fl_experiment.py` calls each
institution's real `flwr.client.NumPyClient` directly in a manual FedAvg loop,
rather than through `flwr.simulation.start_simulation()` (which pulls in Ray).
This keeps the experiment dependency-light and reliable on an ordinary laptop
while still using genuine Flower `Client` objects, real local-only training per
partition, and real weighted weight aggregation — the actual mechanism your
report needs to demonstrate (see the comment at the top of that file). This is
an explicit, defensible scoping choice, in the same spirit as the feasibility
doc's own advice to hand-write the rule engine rather than pull in a rules DSL.

---

## 4. Run the app (two terminals, both from the project root, both with `venv` active)

**Terminal 1 — backend:**
```bash
uvicorn app.backend.main:app --reload --port 8000
```
Check it worked: open http://127.0.0.1:8000/docs — you should see the
interactive API docs.

**Terminal 2 — frontend:**
```bash
streamlit run app/frontend/Home.py
```
This opens http://localhost:8501 in your browser automatically.

---

## 5. Running the live demo (matches Section 10 of your feasibility doc)

1. Go to the **🧑‍🎓 Learner** page. Enter a Learner ID (e.g. `L001`) and name.
2. Type a goal, e.g. *"Teach me Python for my DBMS elective"*, optionally tick
   a couple of concepts you already know.
3. Click **Submit goal / Update plan** — watch the pipeline trace panel. With a
   fresh learner this will usually get **flagged** by rule R2 (max new concepts
   per cycle) — that's expected and is your neuro-symbolic verification working,
   not a bug. Note the XAI explanation shown.
4. Go to the **🧑‍🏫 Educator** page — the same flagged item is waiting there
   with the same explanation. Click **Approve** (or try **Modify**/**Reject**).
5. Back on the **Learner** page, your active plan now reflects the educator's
   decision. Study the first concept's content, answer the check question.
6. Click **Check for gaps & re-plan** — if you answered wrong, you'll see a gap
   detected and a re-plan explanation naming the specific evidence (your recent
   scores + the rule that applied).
7. Visit **🕸️ Knowledge Graph** to see the concept/prerequisite graph rendered,
   with your mastered concepts highlighted green.
8. Visit **📊 Federated Learning** for the FL-vs-centralized accuracy chart —
   present this as evidence rather than something to click through live
   (per Section 10's closing note).

---

## 6. Evaluation metrics (Section 17)

```bash
python -m app.evaluate
```
Runs: the verification self-test (deliberately-bad plans get caught), a
determinism check, personalization diversity across your synthetic learners,
curriculum-alignment pass/flag rate, HITL decision counts, and a naive
"LLM-only" baseline comparison. Copy this JSON output into your evaluation
chapter.

---

## 7. Project structure

```
app/
  db.py                    SQLite schema + CRUD helpers
  kg.py                    Knowledge Graph (NetworkX) — seed course/concepts/rules
  verification.py          Neuro-symbolic rule-checking layer
  explain.py                XAI template engine
  orchestrator.py           Agent Coordination — ties everything together
  evaluate.py                Section 17 metrics
  agents/
    learner_understanding.py
    curriculum_planning.py
    content_generation.py
    progress_evaluation.py
  backend/
    main.py                 FastAPI REST layer
  frontend/
    Home.py                 Streamlit entry point
    api_client.py            requests wrapper around the backend
    pages/
      1_Learner.py
      2_Educator.py
      3_Knowledge_Graph.py
      4_Federated_Learning.py
  data/
    oulad_prep.py            OULAD -> risk-model features, partitioned by institution
    generate_synthetic.py    Synthetic demo learners (Faker)
  risk_model/
    train_baseline.py         Centralized baseline classifier
  fl_experiment/
    model.py                  NumPy logistic regression (get/set params for FedAvg)
    client.py                 Flower NumPyClient per institution
    run_fl_experiment.py       Manual FedAvg loop + results chart
data/oulad/                  Your raw OULAD CSVs
db_files/                    All generated artifacts (SQLite db, KG json, charts, models)
```

## 8. What's real vs simulated (for your report — mirrors Section 18)

| Component | Implemented | Claim | Don't claim |
|---|---|---|---|
| Federated Learning | 4 real OULAD partitions (by `code_module`), real Flower `NumPyClient`, real FedAvg | Working FL prototype demonstrating collaborative improvement without centralizing raw data | Production multi-site deployment |
| XAI | Template-based, built from real rule/evidence structures | Transparent, evidence-grounded explanations | SHAP/LIME or post-hoc interpretability on the plan pathway |
| Neuro-Symbolic | Explicit rules checked via a Python evaluator over the KG | Symbolic verification layer combining data-driven agent output with explicit rules | A learned/differentiable neuro-symbolic inference engine |
| HITL | Queue + Approve/Modify/Reject, logged and applied back to the live plan | Functioning governance mechanism with logged educator authority | Validation with real educators at scale |
| "Institutions" | OULAD's `code_module` field as 4 partitions | Legitimate FL partitioning of real records | Physically separate real organisations |

## 9. If you get stuck

- **"Can't reach the backend API"** in Streamlit → make sure Terminal 1
  (`uvicorn ...`) is still running.
- **`ModuleNotFoundError: No module named 'app'`** → run all commands from the
  project root (the folder containing `app/`, `data/`, `requirements.txt`), not
  from inside `app/`.
- **Port already in use** → change `--port 8000` (backend) or add
  `--server.port 8502` (Streamlit) to use a different port, and update
  `BASE_URL` in `app/frontend/api_client.py` if you changed the backend port.
