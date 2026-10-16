"""
db.py — SQLite storage layer.

Single source of truth for all tables used by the app:
  learners, profiles, answers, plans, hitl_decisions, institution_partitions

Uses plain sqlite3 (no ORM) so it's easy to read/explain in a viva.
"""
import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db_files", "app.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_session():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS learners (
    learner_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
    learner_id TEXT PRIMARY KEY,
    goal_text TEXT,
    goal_course TEXT,
    known_concepts TEXT,        -- JSON list of concept_ids
    mastery TEXT,                -- JSON dict concept_id -> mastery score 0..1
    preferences TEXT,            -- JSON dict
    updated_at TEXT,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
);

CREATE TABLE IF NOT EXISTS plans (
    plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL,
    concept_order TEXT NOT NULL,   -- JSON ordered list of concept_ids
    status TEXT NOT NULL,          -- 'active' | 'superseded' | 'pending_review'
    verification_status TEXT,      -- 'pass' | 'flagged'
    verification_reason TEXT,
    explanation TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
);

CREATE TABLE IF NOT EXISTS answers (
    answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    question TEXT,
    correct_answer TEXT,
    given_answer TEXT,
    is_correct INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
);

CREATE TABLE IF NOT EXISTS hitl_decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL,
    proposal_json TEXT NOT NULL,     -- proposed plan + evidence
    explanation TEXT,
    verification_reason TEXT,
    decision TEXT,                    -- 'pending' | 'approved' | 'modified' | 'rejected'
    modified_plan_json TEXT,
    educator_note TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
);

CREATE TABLE IF NOT EXISTS institution_partitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id TEXT NOT NULL,
    learner_id TEXT NOT NULL,
    source TEXT NOT NULL   -- 'synthetic' | 'oulad'
);
"""


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with db_session() as conn:
        conn.executescript(SCHEMA)


def now():
    return datetime.utcnow().isoformat()


# ---------- Learners / profiles ----------

def upsert_learner(learner_id, name):
    with db_session() as conn:
        conn.execute(
            "INSERT INTO learners (learner_id, name, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(learner_id) DO NOTHING",
            (learner_id, name, now()),
        )


def save_profile(learner_id, goal_text, goal_course, known_concepts, mastery, preferences=None):
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO profiles (learner_id, goal_text, goal_course, known_concepts, mastery, preferences, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(learner_id) DO UPDATE SET
                goal_text=excluded.goal_text,
                goal_course=excluded.goal_course,
                known_concepts=excluded.known_concepts,
                mastery=excluded.mastery,
                preferences=excluded.preferences,
                updated_at=excluded.updated_at
            """,
            (
                learner_id,
                goal_text,
                goal_course,
                json.dumps(known_concepts),
                json.dumps(mastery),
                json.dumps(preferences or {}),
                now(),
            ),
        )


def get_profile(learner_id):
    with db_session() as conn:
        row = conn.execute("SELECT * FROM profiles WHERE learner_id=?", (learner_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["known_concepts"] = json.loads(d["known_concepts"] or "[]")
        d["mastery"] = json.loads(d["mastery"] or "{}")
        d["preferences"] = json.loads(d["preferences"] or "{}")
        return d


def list_learners():
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM learners ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


# ---------- Plans ----------

def save_plan(learner_id, concept_order, status, verification_status, verification_reason, explanation):
    with db_session() as conn:
        conn.execute("UPDATE plans SET status='superseded' WHERE learner_id=? AND status='active'", (learner_id,))
        cur = conn.execute(
            """
            INSERT INTO plans (learner_id, concept_order, status, verification_status,
                                verification_reason, explanation, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (learner_id, json.dumps(concept_order), status, verification_status,
             verification_reason, explanation, now()),
        )
        return cur.lastrowid


def get_active_plan(learner_id):
    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM plans WHERE learner_id=? AND status='active' ORDER BY created_at DESC LIMIT 1",
            (learner_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["concept_order"] = json.loads(d["concept_order"])
        return d


def get_plan(plan_id):
    with db_session() as conn:
        row = conn.execute("SELECT * FROM plans WHERE plan_id=?", (plan_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["concept_order"] = json.loads(d["concept_order"])
        return d


# ---------- Answers ----------

def save_answer(learner_id, concept_id, question, correct_answer, given_answer, is_correct):
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO answers (learner_id, concept_id, question, correct_answer, given_answer, is_correct, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (learner_id, concept_id, question, correct_answer, given_answer, int(is_correct), now()),
        )


def get_answers(learner_id, concept_id=None):
    with db_session() as conn:
        if concept_id:
            rows = conn.execute(
                "SELECT * FROM answers WHERE learner_id=? AND concept_id=? ORDER BY created_at",
                (learner_id, concept_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM answers WHERE learner_id=? ORDER BY created_at", (learner_id,)
            ).fetchall()
        return [dict(r) for r in rows]


# ---------- HITL ----------

def enqueue_hitl(learner_id, proposal, explanation, verification_reason):
    with db_session() as conn:
        cur = conn.execute(
            """
            INSERT INTO hitl_decisions (learner_id, proposal_json, explanation, verification_reason, decision, created_at)
            VALUES (?, ?, ?, ?, 'pending', ?)
            """,
            (learner_id, json.dumps(proposal), explanation, verification_reason, now()),
        )
        return cur.lastrowid


def get_pending_hitl():
    with db_session() as conn:
        rows = conn.execute(
            "SELECT * FROM hitl_decisions WHERE decision='pending' ORDER BY created_at"
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["proposal_json"] = json.loads(d["proposal_json"])
            out.append(d)
        return out


def decide_hitl(decision_id, decision, modified_plan=None, educator_note=""):
    with db_session() as conn:
        conn.execute(
            """
            UPDATE hitl_decisions
            SET decision=?, modified_plan_json=?, educator_note=?, decided_at=?
            WHERE decision_id=?
            """,
            (decision, json.dumps(modified_plan) if modified_plan else None, educator_note, now(), decision_id),
        )
        row = conn.execute("SELECT * FROM hitl_decisions WHERE decision_id=?", (decision_id,)).fetchone()
        return dict(row)


def get_hitl_history():
    with db_session() as conn:
        rows = conn.execute(
            "SELECT * FROM hitl_decisions WHERE decision != 'pending' ORDER BY decided_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
