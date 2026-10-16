"""
main.py — FastAPI backend (Level 2+).

Thin REST layer over the Orchestrator. The Streamlit frontend talks to this
instead of calling agent functions directly, so frontend/backend can run and be
demoed as two separate processes -- and examiners can poke the endpoints
directly via the auto-generated docs at /docs.

Run:  uvicorn app.backend.main:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app import db, kg as kg_mod
from app.orchestrator import Orchestrator

app = FastAPI(title="Agentic AI for Personalized Learning — API", version="0.1")

db.init_db()
orchestrator = Orchestrator()


# ---------- request/response models ----------

class GoalRequest(BaseModel):
    learner_id: str
    learner_name: str
    goal_text: str
    known_concept_names: List[str] = []


class AnswerRequest(BaseModel):
    learner_id: str
    concept_id: str
    question: str
    correct_answer: str
    given_answer: str


class HitlDecisionRequest(BaseModel):
    decision_id: int
    decision: str  # 'approved' | 'modified' | 'rejected'
    modified_concept_order: Optional[List[str]] = None
    note: str = ""


# ---------- KG / meta ----------

@app.get("/kg/concepts")
def kg_concepts():
    g = orchestrator.g
    course_id = kg_mod.SEED_COURSE
    concepts = kg_mod.get_concepts_for_course(g, course_id)
    return [{"concept_id": c, "name": kg_mod.concept_name(g, c)} for c in concepts]


@app.get("/kg/rules")
def kg_rules():
    g = orchestrator.g
    return kg_mod.get_rules_for_course(g, kg_mod.SEED_COURSE)


# ---------- Learner flow ----------

@app.post("/goal")
def submit_goal(req: GoalRequest):
    result = orchestrator.start_or_update_goal(
        req.learner_id, req.learner_name, req.goal_text, req.known_concept_names
    )
    return result


@app.get("/plan/{learner_id}")
def active_plan(learner_id: str):
    plan = db.get_active_plan(learner_id)
    if not plan:
        raise HTTPException(404, "No active plan for this learner yet")
    return plan


@app.get("/content/{concept_id}")
def content(concept_id: str):
    return orchestrator.get_content(concept_id)


@app.post("/answer")
def submit_answer(req: AnswerRequest):
    return orchestrator.submit_answer(
        req.learner_id, req.concept_id, req.question, req.correct_answer, req.given_answer
    )


@app.post("/replan/{learner_id}")
def replan(learner_id: str):
    return orchestrator.replan_if_needed(learner_id)


@app.get("/profile/{learner_id}")
def profile(learner_id: str):
    p = db.get_profile(learner_id)
    if not p:
        raise HTTPException(404, "No profile for this learner yet")
    return p


@app.get("/learners")
def learners():
    return db.list_learners()


# ---------- Educator / HITL flow ----------

@app.get("/hitl/pending")
def hitl_pending():
    return db.get_pending_hitl()


@app.get("/hitl/history")
def hitl_history():
    return db.get_hitl_history()


@app.post("/hitl/decide")
def hitl_decide(req: HitlDecisionRequest):
    if req.decision not in {"approved", "modified", "rejected"}:
        raise HTTPException(400, "decision must be approved | modified | rejected")
    if req.decision == "modified" and not req.modified_concept_order:
        raise HTTPException(400, "modified_concept_order is required when decision='modified'")
    return orchestrator.apply_hitl_decision(
        req.decision_id, req.decision, req.modified_concept_order, req.note
    )


@app.get("/health")
def health():
    return {"status": "ok"}
