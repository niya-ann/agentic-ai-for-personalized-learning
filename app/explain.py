"""
explain.py — the XAI layer.

Renders the evidence the pipeline already computed (rule matches, gaps, scores)
into a short, structured, human-readable explanation. No separate inference step:
the explanation IS the reasoning, just rendered as text (see Section 14 of the
feasibility doc for why this is the correct scope, not a simplification).
"""
from app import kg as kg_mod


def explain_plan_pass(concept_order, g):
    names = [kg_mod.concept_name(g, c) for c in concept_order]
    return (
        f"Plan verified against all curriculum rules. Order respects every prerequisite "
        f"edge in the Knowledge Graph: {' -> '.join(names)}."
    )


def explain_flag(verification_result, g):
    """Build an explanation string for a flagged plan/recommendation."""
    details = verification_result.get("details", {})
    parts = []

    if "prereq_violation" in details:
        later, missing = details["prereq_violation"]
        parts.append(
            f"'{kg_mod.concept_name(g, later)}' was scheduled before its required prerequisite "
            f"'{kg_mod.concept_name(g, missing)}', which violates curriculum rule R1 "
            f"(prerequisites must precede dependents)."
        )
    if "overload" in details:
        o = details["overload"]
        parts.append(
            f"The plan introduces {o['count']} new concepts at once, exceeding the institutional "
            f"limit of {o['limit']} per cycle (rule R2)."
        )
    if "missing_mandatory" in details:
        parts.append(
            f"The plan omits the mandatory concept "
            f"'{kg_mod.concept_name(g, details['missing_mandatory'])}' required by rule R3."
        )
    if not parts:
        parts.append(verification_result.get("reason", "A curriculum rule was not satisfied."))

    return " ".join(parts)


def explain_replan_from_gap(concept_id, scores, threshold, g):
    """Explanation for why a re-plan was triggered by a detected learning gap."""
    name = kg_mod.concept_name(g, concept_id)
    score_str = ", ".join(f"{s:.0%}" for s in scores)
    return (
        f"Recommended because your recent answers on '{name}' scored {score_str} "
        f"(below the {threshold:.0%} mastery threshold), and '{name}' is a prerequisite "
        f"for later concepts per the curriculum graph. Extra practice on '{name}' has been "
        f"inserted before continuing."
    )


def build_evidence_trail(concept_id, mastery, gap_threshold, verification_result):
    """Structured evidence dict shown alongside the natural-language explanation
    (learner UI / educator dashboard both render this)."""
    return {
        "concept": concept_id,
        "mastery_score": mastery,
        "gap_threshold": gap_threshold,
        "gap_detected": mastery < gap_threshold,
        "verification_status": verification_result["status"],
        "fired_rules": verification_result["fired_rules"],
    }
