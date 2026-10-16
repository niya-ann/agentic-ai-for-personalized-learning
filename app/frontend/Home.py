"""
Home.py — Streamlit multipage app entry point.

Run:  streamlit run app/frontend/Home.py
(the FastAPI backend must already be running on port 8000 — see README)
"""
import streamlit as st

st.set_page_config(page_title="Agentic AI for Personalized Learning", page_icon="🎓", layout="wide")

st.title("🎓 Agentic AI for Personalized Learning")
st.caption("Explainable, Neuro-Symbolic, Federated Learning Framework — Level 2 Prototype")

st.markdown("""
This prototype demonstrates the full pipeline from your Review-1 report, at the scope
described in the feasibility document:

- **Multi-agent orchestration** — 4 cooperating agents + a controller, with a re-plan loop
- **Knowledge Graph** — NetworkX graph of concepts, prerequisites, skills, resources, rules
- **Neuro-Symbolic Verification** — rule-checking layer that flags plans violating curriculum rules
- **XAI** — evidence-grounded, template-based explanations for every recommendation
- **Human-in-the-Loop** — an Educator dashboard with Approve / Modify / Reject
- **Federated Learning** — a real FedAvg experiment across 4 simulated institutions
  (partitioned from the OULAD dataset by course module), compared to a centralized baseline

### Use the sidebar to navigate:
- **🧑‍🎓 Learner** — state a goal, get a curriculum-checked plan, study content, take checks
- **🧑‍🏫 Educator** — review any AI recommendation flagged by the verification layer
- **🕸️ Knowledge Graph** — see the concept/prerequisite graph
- **📊 Federated Learning** — the FL vs centralized accuracy results and chart

---
**What's real vs simulated in this build** (be explicit about this in your report, Section 18):
- Real: the KG, rule engine, agent orchestration, SQLite storage, HITL workflow,
  the FedAvg training loop (genuine Flower `NumPyClient` objects, real per-institution
  local training, real weighted weight aggregation), and the explanation generation logic.
- Simulated: "multiple institutions" (partitions of the OULAD dataset by course module,
  clearly labelled as such, not physically separate organisations); "Learner" and "Educator"
  are two roles in the same app rather than separate organisations.
""")

st.info("Start here → pick **🧑‍🎓 Learner** in the sidebar to run the live demo scenario "
        "from Section 10 of your feasibility document.")
