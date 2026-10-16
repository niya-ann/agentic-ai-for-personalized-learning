"""
generate_synthetic.py — populates the app's SQLite with synthetic demo learners
(Section 16). This is separate from the OULAD-derived risk-model data: these
learners exist so the Learner/Educator UI has more than one person to demo with,
and so Section 17's "personalization diversity" metric has something to measure.

Run:  python -m app.data.generate_synthetic --n 60
"""
import argparse
import random
from faker import Faker

from app import db, kg as kg_mod
from app.agents import learner_understanding
from app.orchestrator import Orchestrator

fake = Faker()

GOAL_TEMPLATES = [
    "Teach me Python for my DBMS elective",
    "I want to learn Python programming from scratch",
    "Help me get better at Python for my data science course",
    "I need Python for an automation project",
    "Prepare me for a Python coding interview",
]


def generate(n=60, seed=42):
    random.seed(seed)
    Faker.seed(seed)
    db.init_db()
    g = kg_mod.load_graph()
    all_concept_names = [kg_mod.concept_name(g, c) for c in kg_mod.get_concepts_for_course(g, kg_mod.SEED_COURSE)]

    created = []
    orch = Orchestrator()
    for i in range(n):
        learner_id = f"SYN{i:04d}"
        name = fake.name()
        goal_text = random.choice(GOAL_TEMPLATES)
        # a synthetic learner already knows a random-sized random subset of concepts
        n_known = random.choices([0, 1, 2, 3, 4], weights=[30, 25, 20, 15, 10])[0]
        known = random.sample(all_concept_names, k=min(n_known, len(all_concept_names)))

        learner_understanding.understand_learner(learner_id, name, goal_text, known, g)
        orch._plan_verify_apply(learner_id, db.get_profile(learner_id), new_concepts=None)

        # simulate a few prior answers so mastery/gap data isn't all zero
        from app.agents import content_generation, progress_evaluation
        n_answers = random.randint(0, 4)
        concepts_to_answer = random.sample(all_concept_names, k=min(n_answers, len(all_concept_names)))
        for cname in concepts_to_answer:
            cid = [c for c in kg_mod.get_concepts_for_course(g, kg_mod.SEED_COURSE)
                   if kg_mod.concept_name(g, c) == cname][0]
            content = content_generation.generate_content(cid, g)
            correct = random.random() < 0.55
            given = content["answer"] if correct else "I don't know"
            progress_evaluation.evaluate_answer(learner_id, cid, content["question"], content["answer"], given, correct)

        # record which "institution" this synthetic learner notionally belongs to
        # (kept separate from the OULAD-based FL demo; useful only for narrative consistency)
        inst = random.choice(["Inst_A", "Inst_B", "Inst_C"])
        with db.db_session() as conn:
            conn.execute(
                "INSERT INTO institution_partitions (institution_id, learner_id, source) VALUES (?, ?, 'synthetic')",
                (inst, learner_id),
            )

        created.append(learner_id)

    print(f"Generated {len(created)} synthetic learners: {created[0]} .. {created[-1]}")
    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    args = parser.parse_args()
    generate(n=args.n)
