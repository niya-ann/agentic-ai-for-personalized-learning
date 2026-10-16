"""
pages/4_Federated_Learning.py — presents the FL experiment as evidence
(Section 10's closing tab; FL isn't naturally something to click through live).
"""
import os
import json
import streamlit as st

st.set_page_config(page_title="Federated Learning", page_icon="📊", layout="wide")
st.title("📊 Federated Learning Experiment")

st.markdown("""
Institutions are simulated as isolated data partitions on a single machine, using
Flower's `NumPyClient` abstraction and real FedAvg aggregation. Partitions come from
the **OULAD dataset**, split by `code_module` (course) — a genuine, non-overlapping
partition of real student records, not an artificial split of one pool.
""")

HERE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "db_files")
results_path = os.path.join(DB_DIR, "fl_results.json")
chart_path = os.path.join(DB_DIR, "fl_accuracy_chart.png")
baseline_path = os.path.join(DB_DIR, "baseline_results.json")

if not os.path.exists(results_path):
    st.warning("No FL results yet. Run this first in a terminal:\n\n"
               "`python -m app.fl_experiment.run_fl_experiment`")
else:
    with open(results_path) as f:
        results = json.load(f)

    col1, col2, col3 = st.columns(3)
    col1.metric("Institutions (simulated)", results["n_institutions"])
    col2.metric("Final federated accuracy", f"{results['final_federated_accuracy']:.1%}")
    col3.metric("Final centralized accuracy", f"{results['final_centralized_accuracy']:.1%}")

    st.image(chart_path, caption="Federated (FedAvg) vs Centralized baseline accuracy per round")

    st.caption(
        "Institutions used: " + ", ".join(results["institutions"]) +
        f" · {results['n_rounds']} communication rounds."
    )

    with st.expander("Raw per-round results"):
        st.json(results)

if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baseline = json.load(f)
    st.subheader("Reference: single centrally-trained scikit-learn model")
    st.json(baseline)
