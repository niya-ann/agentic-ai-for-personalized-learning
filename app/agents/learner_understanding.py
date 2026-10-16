"""
Learner Understanding Agent.

Reads the learner's stated goal + self-reported known concepts + any prior scores,
and writes/updates a structured LearnerProfile (SQLite row).

Goal -> course/concept matching is keyword/rule-based (Section 5/8): enough for a
prototype with one seed course, no embedding model required.
"""
from app import db, kg as kg_mod

GOAL_KEYWORDS = {
    "python_basics": ["python", "dbms", "programming", "coding", "loops", "functions"],
}


def match_goal_to_course(goal_text, g):
    text = goal_text.lower()
    best_course, best_score = None, 0
    for node, data in g.nodes(data=True):
        if data.get("type") != "course":
            continue
        keywords = GOAL_KEYWORDS.get(node, [node.replace("_", " ")])
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_course, best_score = node, score
    # fallback: only one seed course in this prototype, so default to it
    return best_course or kg_mod.SEED_COURSE


def understand_learner(learner_id, learner_name, goal_text, known_concept_names, g):
    """
    known_concept_names: list of display names the learner self-reports knowing
                          (as picked from a multi-select in the UI)
    Returns a LearnerProfile dict and persists it to SQLite.
    """
    db.upsert_learner(learner_id, learner_name)

    course_id = match_goal_to_course(goal_text, g)
    all_concepts = kg_mod.get_concepts_for_course(g, course_id)

    name_to_id = {kg_mod.concept_name(g, c): c for c in all_concepts}
    known_ids = [name_to_id[n] for n in known_concept_names if n in name_to_id]

    prior_answers = db.get_answers(learner_id)
    mastery = {}
    for c in all_concepts:
        if c in known_ids:
            mastery[c] = 1.0
        else:
            concept_answers = [a for a in prior_answers if a["concept_id"] == c]
            if concept_answers:
                mastery[c] = sum(a["is_correct"] for a in concept_answers) / len(concept_answers)
            else:
                mastery[c] = 0.0

    db.save_profile(
        learner_id=learner_id,
        goal_text=goal_text,
        goal_course=course_id,
        known_concepts=known_ids,
        mastery=mastery,
    )

    return db.get_profile(learner_id)
