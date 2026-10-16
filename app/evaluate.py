"""
evaluate.py — implements the Section 17 evaluation metrics that are cheap to
compute automatically: verification self-test, determinism, plan diversity,
and a naive "LLM-only" baseline comparison.

Run:  python -m app.evaluate
"""
import json
from app import db, kg as kg_mod, verification
from app.orchestrator import Orchestrator


def metric_verification_catches_bad_plans(g):
    result = verification.run_self_test(g, kg_mod.SEED_COURSE)
    return {"caught_bad_plan": result["caught_bad_plan"], "detail": result}


def metric_determinism(orch):
    """Same profile + same goal, run twice -> same plan."""
    r1 = orch.start_or_update_goal("EVAL_DET", "Det Test", "Teach me Python", ["Variables & Data Types"])
    r2 = orch.start_or_update_goal("EVAL_DET", "Det Test", "Teach me Python", ["Variables & Data Types"])
    same = r1["concept_order"] == r2["concept_order"]
    return {"deterministic": same}


def metric_personalization_diversity():
    """% of synthetic learners' plans that differ from each other."""
    with db.db_session() as conn:
        rows = conn.execute(
            "SELECT learner_id, concept_order FROM plans WHERE status IN ('active','pending_review') "
            "AND learner_id LIKE 'SYN%'"
        ).fetchall()
    plans = [r["concept_order"] for r in rows]
    if len(plans) < 2:
        return {"note": "run generate_synthetic first", "n_plans": len(plans)}
    unique = len(set(plans))
    return {"n_plans": len(plans), "n_unique": unique, "diversity_ratio": round(unique / len(plans), 3)}


def metric_curriculum_alignment():
    """% of plans that passed verification without needing a flag."""
    with db.db_session() as conn:
        rows = conn.execute("SELECT verification_status FROM plans").fetchall()
    total = len(rows)
    if total == 0:
        return {"note": "no plans yet"}
    passed = sum(1 for r in rows if r["verification_status"] == "pass")
    return {"total_plans": total, "passed": passed, "flagged": total - passed,
             "pass_rate": round(passed / total, 3)}


def metric_hitl_effectiveness():
    with db.db_session() as conn:
        rows = conn.execute("SELECT decision FROM hitl_decisions WHERE decision != 'pending'").fetchall()
    total = len(rows)
    if total == 0:
        return {"note": "no HITL decisions yet"}
    counts = {}
    for r in rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    return {"total_decisions": total, **counts}


def naive_llm_only_baseline(goal_text="Teach me Python for my DBMS elective"):
    """A one-shot 'baseline' with no KG/rules/HITL/verification: it just returns
    concepts in an arbitrary (non-prerequisite-aware) order, to contrast against
    the framework's verified output (Section 17's baseline comparison)."""
    import random
    g = kg_mod.load_graph()
    concepts = kg_mod.get_concepts_for_course(g, kg_mod.SEED_COURSE)
    random.seed(1)
    naive_order = concepts.copy()
    random.shuffle(naive_order)
    verify_result = verification.verify(naive_order, kg_mod.SEED_COURSE, g)
    return {
        "naive_order": naive_order,
        "would_pass_verification": verify_result["status"] == "pass",
        "note": "Unlike the framework's planner, a naive/LLM-only baseline has no "
                "prerequisite-awareness and is not guaranteed to pass the same checks.",
    }


def run_all():
    db.init_db()
    g = kg_mod.load_graph()
    orch = Orchestrator()

    results = {
        "verification_self_test": metric_verification_catches_bad_plans(g),
        "determinism": metric_determinism(orch),
        "personalization_diversity": metric_personalization_diversity(),
        "curriculum_alignment": metric_curriculum_alignment(),
        "hitl_effectiveness": metric_hitl_effectiveness(),
        "naive_baseline_comparison": naive_llm_only_baseline(),
    }
    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    run_all()
