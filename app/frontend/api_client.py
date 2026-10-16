"""
api_client.py — thin wrapper the Streamlit pages use to talk to the FastAPI
backend (Section 9's Streamlit <-> FastAPI <-> Orchestrator architecture).

Requires the backend to be running:
    uvicorn app.backend.main:app --reload --port 8000
"""
import requests
import streamlit as st

BASE_URL = "http://127.0.0.1:8000"


def _get(path, **kwargs):
    try:
        r = requests.get(f"{BASE_URL}{path}", timeout=10, **kwargs)
        if r.status_code >= 400:
            return None, r.json().get("detail", r.text)
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "backend_unreachable"


def _post(path, json_body):
    try:
        r = requests.post(f"{BASE_URL}{path}", json=json_body, timeout=10)
        if r.status_code >= 400:
            return None, r.json().get("detail", r.text)
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "backend_unreachable"


def backend_warning_if_unreachable(error):
    if error == "backend_unreachable":
        st.error(
            "Can't reach the backend API. Start it first in a separate terminal:\n\n"
            "`uvicorn app.backend.main:app --reload --port 8000`"
        )
        st.stop()
    elif error:
        st.error(f"API error: {error}")
        st.stop()


def get_concepts():
    data, err = _get("/kg/concepts")
    backend_warning_if_unreachable(err)
    return data


def get_rules():
    data, err = _get("/kg/rules")
    backend_warning_if_unreachable(err)
    return data


def submit_goal(learner_id, learner_name, goal_text, known_concept_names):
    data, err = _post("/goal", {
        "learner_id": learner_id, "learner_name": learner_name,
        "goal_text": goal_text, "known_concept_names": known_concept_names,
    })
    backend_warning_if_unreachable(err)
    return data


def get_active_plan(learner_id):
    data, err = _get(f"/plan/{learner_id}")
    if err == "backend_unreachable":
        backend_warning_if_unreachable(err)
    return data  # may be None if 404 (no plan yet) -- caller handles that


def get_content(concept_id):
    data, err = _get(f"/content/{concept_id}")
    backend_warning_if_unreachable(err)
    return data


def submit_answer(learner_id, concept_id, question, correct_answer, given_answer):
    data, err = _post("/answer", {
        "learner_id": learner_id, "concept_id": concept_id, "question": question,
        "correct_answer": correct_answer, "given_answer": given_answer,
    })
    backend_warning_if_unreachable(err)
    return data


def replan(learner_id):
    data, err = _post(f"/replan/{learner_id}", {})
    backend_warning_if_unreachable(err)
    return data


def get_profile(learner_id):
    data, err = _get(f"/profile/{learner_id}")
    return data  # None on 404 is expected for a brand-new learner


def list_learners():
    data, err = _get("/learners")
    backend_warning_if_unreachable(err)
    return data


def hitl_pending():
    data, err = _get("/hitl/pending")
    backend_warning_if_unreachable(err)
    return data


def hitl_history():
    data, err = _get("/hitl/history")
    backend_warning_if_unreachable(err)
    return data


def hitl_decide(decision_id, decision, modified_concept_order=None, note=""):
    data, err = _post("/hitl/decide", {
        "decision_id": decision_id, "decision": decision,
        "modified_concept_order": modified_concept_order, "note": note,
    })
    backend_warning_if_unreachable(err)
    return data
