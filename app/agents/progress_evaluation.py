"""
Progress Evaluation Agent.

Scores the learner's answer, updates mastery (exponential moving average),
and flags concepts as learning gaps if mastery stays below threshold.
"""
from app import db
from app.agents.curriculum_planning import MASTERY_THRESHOLD

EMA_ALPHA = 0.6  # weight on the newest observation


def evaluate_answer(learner_id, concept_id, question, correct_answer, given_answer, is_correct):
    db.save_answer(learner_id, concept_id, question, correct_answer, given_answer, is_correct)

    profile = db.get_profile(learner_id)
    old_mastery = profile["mastery"].get(concept_id, 0.0)
    new_mastery = EMA_ALPHA * (1.0 if is_correct else 0.0) + (1 - EMA_ALPHA) * old_mastery
    profile["mastery"][concept_id] = new_mastery

    db.save_profile(
        learner_id=learner_id,
        goal_text=profile["goal_text"],
        goal_course=profile["goal_course"],
        known_concepts=profile["known_concepts"],
        mastery=profile["mastery"],
    )

    gap = new_mastery < MASTERY_THRESHOLD
    return {
        "concept_id": concept_id,
        "old_mastery": old_mastery,
        "new_mastery": new_mastery,
        "is_gap": gap,
    }


def detect_gaps(learner_id):
    """A gap is a concept the learner has actually attempted (has >=1 answer on
    record) whose mastery is still below threshold -- untouched concepts (mastery
    defaulting to 0) don't count as 'gaps' since nothing has been assessed yet."""
    profile = db.get_profile(learner_id)
    if not profile:
        return []
    attempted = {a["concept_id"] for a in db.get_answers(learner_id)}
    return [
        c for c, score in profile["mastery"].items()
        if c in attempted and score < MASTERY_THRESHOLD
    ]


def recent_scores(learner_id, concept_id, n=3):
    answers = db.get_answers(learner_id, concept_id)
    return [a["is_correct"] for a in answers[-n:]]
