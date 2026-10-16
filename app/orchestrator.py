"""
orchestrator.py — the controller tying the four agents together.

Implements the execution flow from Section 1 of the feasibility doc:
  1. Learner submits goal            -> Learner Understanding Agent
  2. Build plan                      -> Curriculum & Planning Agent
  3. Verify plan                     -> verification.py
  4. If flagged -> HITL queue (with XAI explanation); else -> active plan
  5/6/7. Content + answer + scoring  -> Content Generation + Progress Evaluation Agents
  8. If gaps -> re-plan (loop back to step 2), again through verify + XAI + HITL
"""
from app import db, kg as kg_mod, verification, explain
from app.agents import learner_understanding, curriculum_planning, progress_evaluation


class Orchestrator:
    def __init__(self):
        self.g = kg_mod.load_graph()

    # ---- Step 1-4: goal intake -> plan -> verify -> (HITL or active) ----
    def start_or_update_goal(self, learner_id, learner_name, goal_text, known_concept_names):
        profile = learner_understanding.understand_learner(
            learner_id, learner_name, goal_text, known_concept_names, self.g
        )
        return self._plan_verify_apply(learner_id, profile, new_concepts=None)

    def _plan_verify_apply(self, learner_id, profile, new_concepts):
        ordered, course_id = curriculum_planning.build_plan(profile, self.g)
        result = verification.verify(ordered, course_id, self.g, new_concepts_since_last=new_concepts)

        if result["status"] == "pass":
            explanation = explain.explain_plan_pass(ordered, self.g)
            plan_id = db.save_plan(learner_id, ordered, "active", "pass", None, explanation)
            return {
                "status": "active",
                "plan_id": plan_id,
                "concept_order": ordered,
                "explanation": explanation,
                "verification": result,
            }
        else:
            explanation = explain.explain_flag(result, self.g)
            plan_id = db.save_plan(learner_id, ordered, "pending_review", "flagged", result["reason"], explanation)
            hitl_id = db.enqueue_hitl(
                learner_id,
                proposal={"plan_id": plan_id, "concept_order": ordered, "course_id": course_id},
                explanation=explanation,
                verification_reason=result["reason"],
            )
            return {
                "status": "pending_review",
                "plan_id": plan_id,
                "hitl_id": hitl_id,
                "concept_order": ordered,
                "explanation": explanation,
                "verification": result,
            }

    # ---- Steps 5-7: content + scoring ----
    def get_content(self, concept_id):
        from app.agents import content_generation
        return content_generation.generate_content(concept_id, self.g)

    def submit_answer(self, learner_id, concept_id, question, correct_answer, given_answer):
        from app.agents import content_generation
        is_correct = content_generation.grade_answer(concept_id, given_answer, correct_answer)
        eval_result = progress_evaluation.evaluate_answer(
            learner_id, concept_id, question, correct_answer, given_answer, is_correct
        )
        return eval_result

    # ---- Step 8: gap-driven re-plan ----
    def replan_if_needed(self, learner_id):
        # Detect learning gaps and perform minimal adaptive repair of the
        # learner's current active plan.
        # The replanning process deliberately starts from the existing active
        # plan rather than rebuilding the entire unmet curriculum.
        gaps = progress_evaluation.detect_gaps(learner_id)

        if not gaps:
            return {
                "replanned": False,
                "gaps": []
            }

        # ---------------------------------------------------------
        # 2. Retrieve learner profile
        # ---------------------------------------------------------
        profile = db.get_profile(learner_id)

        if not profile:
            return {
                "replanned": False,
                "gaps": gaps,
                "status": "error",
                "message": "Learner profile not found."
            }

        # ---------------------------------------------------------
        # 3. Retrieve the CURRENT active plan
        # ---------------------------------------------------------
        active_plan = db.get_active_plan(learner_id)

        if not active_plan:
            # If there is no active plan to repair, safely fall back
            # to normal initial planning.
            ordered, course_id = curriculum_planning.build_plan(
                profile,
                self.g
            )

        else:
            # -----------------------------------------------------
            # 4. Perform MINIMAL adaptive plan repair
            # -----------------------------------------------------
            ordered, course_id = (
                curriculum_planning.build_replan_with_gap_priority(
                    profile,
                    self.g,
                    gaps,
                    active_plan
                )
            )

        # ---------------------------------------------------------
        # 5. Verify the repaired plan
        # ---------------------------------------------------------
        result = verification.verify(
            ordered,
            course_id,
            self.g,
            new_concepts_since_last=gaps
        )

        # ---------------------------------------------------------
        # 6. Generate XAI explanation for the detected gaps
        # ---------------------------------------------------------
        gap_explanations = []

        for gap_c in gaps:
            scores = [
                float(s)
                for s in progress_evaluation.recent_scores(
                    learner_id,
                    gap_c
                )
            ]

            if scores:
                gap_explanations.append(
                    explain.explain_replan_from_gap(
                        gap_c,
                        scores,
                        curriculum_planning.MASTERY_THRESHOLD,
                        self.g
                    )
                )

        # ---------------------------------------------------------
        # 7. Apply automatically if verification passes
        # ---------------------------------------------------------
        if result["status"] == "pass":

            explanation = (
                " ".join(gap_explanations)
                or explain.explain_plan_pass(ordered, self.g)
            )

            plan_id = db.save_plan(
                learner_id,
                ordered,
                "active",
                "pass",
                None,
                explanation
            )

            return {
                "replanned": True,
                "status": "active",
                "plan_id": plan_id,
                "concept_order": ordered,
                "gaps": gaps,
                "explanation": explanation,
                "verification": result,
            }

        # ---------------------------------------------------------
        # 8. Route verification failures to HITL
        # ---------------------------------------------------------
        else:

            explanation = (
                explain.explain_flag(result, self.g)
                + " "
                + " ".join(gap_explanations)
            )

            plan_id = db.save_plan(
                learner_id,
                ordered,
                "pending_review",
                "flagged",
                result.get("reason"),
                explanation
            )

            hitl_id = db.enqueue_hitl(
                learner_id,
                {
                    "plan_id": plan_id,
                    "concept_order": ordered,
                    "gaps": gaps,
                    "verification": result,
                },
                explanation,
                result.get("reason")
            )

            return {
                "replanned": True,
                "status": "pending_review",
                "plan_id": plan_id,
                "hitl_id": hitl_id,
                "concept_order": ordered,
                "gaps": gaps,
                "explanation": explanation,
                "verification": result,
            }

    # ---- HITL decisions applied back to the learner's active plan ----
    def apply_hitl_decision(self, decision_id, decision, modified_concept_order=None, note=""):
        record = db.decide_hitl(decision_id, decision, modified_concept_order, note)
        learner_id = record["learner_id"]
        import json
        proposal = json.loads(record["proposal_json"])

        if decision == "approved":
            db.save_plan(learner_id, proposal["concept_order"], "active", "pass",
                         None, "Approved by educator: " + record["explanation"])
        elif decision == "modified":
            db.save_plan(learner_id, modified_concept_order, "active", "pass",
                         None, "Modified by educator: " + (note or ""))
        # rejected -> previous active plan (if any) stays as-is; nothing new written
        return record
