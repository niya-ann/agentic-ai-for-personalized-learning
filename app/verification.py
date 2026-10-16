"""
verification.py — the Neuro-Symbolic Verification layer.

Takes an agent-generated LearningPlan (ordered list of concept_ids) and checks it
against the explicit CurriculumRule nodes stored in the Knowledge Graph.
Pure conditional logic over structured data -- no learned rule weights.

verify() returns a VerificationResult:
    {
        "status": "pass" | "flagged",
        "reason": str | None,       # human string, used by explain.py
        "fired_rules": [rule_id, ...],
        "details": {...}            # structured evidence for XAI
    }
"""
from app import kg as kg_mod


def verify(concept_order, course_id, g, new_concepts_since_last=None):
    """
    concept_order: ordered list of concept_ids (the candidate plan)
    course_id: which course's rules to check
    g: the loaded NetworkX graph
    new_concepts_since_last: optional list of concept_ids being newly introduced
                              (used by the max-load rule); defaults to concept_order
    """
    rules = kg_mod.get_rules_for_course(g, course_id)
    fired = []
    reasons = []
    details = {}

    new_concepts_since_last = new_concepts_since_last or concept_order

    for rule in rules:
        kind = rule.get("rule_kind")

        if kind == "prerequisite_order":
            violation = _check_prereq_order(concept_order, g)
            if violation:
                fired.append(rule["rule_id"])
                reasons.append(
                    f"{rule['rule_id']}: '{kg_mod.concept_name(g, violation[0])}' appears before its "
                    f"prerequisite '{kg_mod.concept_name(g, violation[1])}'."
                )
                details["prereq_violation"] = violation

        elif kind == "max_load":
            max_new = rule.get("max_new_concepts", 4)
            if len(new_concepts_since_last) > max_new:
                fired.append(rule["rule_id"])
                reasons.append(
                    f"{rule['rule_id']}: plan introduces {len(new_concepts_since_last)} new concepts, "
                    f"exceeding the limit of {max_new}."
                )
                details["overload"] = {"count": len(new_concepts_since_last), "limit": max_new}

        elif kind == "mandatory_concept":
            required = rule.get("concept_id")
            if required not in concept_order:
                fired.append(rule["rule_id"])
                reasons.append(
                    f"{rule['rule_id']}: mandatory concept '{kg_mod.concept_name(g, required)}' is missing "
                    f"from the plan."
                )
                details["missing_mandatory"] = required

    if fired:
        return {
            "status": "flagged",
            "reason": " | ".join(reasons),
            "fired_rules": fired,
            "details": details,
        }
    return {"status": "pass", "reason": None, "fired_rules": [], "details": {}}


def _check_prereq_order(concept_order, g):
    """Return (later_concept, missing_prereq) if some concept appears before a
    prerequisite that the graph says it needs, else None."""
    prereq_g = kg_mod.get_prereq_subgraph(g)
    position = {c: i for i, c in enumerate(concept_order)}
    for c in concept_order:
        for pred in prereq_g.predecessors(c):
            if pred in position and position[pred] > position[c]:
                return (c, pred)
    return None


def run_self_test(g, course_id):
    """Deliberately-bad test plans -> confirm the layer catches them (Section 17 evaluation)."""
    concepts = kg_mod.get_concepts_for_course(g, course_id)
    good_order = kg_mod.topological_concept_order(g, concepts)
    bad_order = list(reversed(good_order))  # almost certainly violates prereqs

    good_result = verify(good_order, course_id, g)
    bad_result = verify(bad_order, course_id, g)
    return {
        "good_plan_result": good_result,
        "bad_plan_result": bad_result,
        "caught_bad_plan": bad_result["status"] == "flagged",
    }


if __name__ == "__main__":
    g = kg_mod.load_graph()
    print(run_self_test(g, kg_mod.SEED_COURSE))
