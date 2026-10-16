"""
pages/1_Learner.py — the learner flow (Section 10 demo script, steps 1-9).

goal -> profile -> plan -> verify (pass/flag) -> content -> answer -> score ->
gap detection -> re-plan
"""
import streamlit as st
from app.frontend import api_client as api

st.set_page_config(page_title="Learner", page_icon="🧑‍🎓", layout="wide")
st.title("🧑‍🎓 Learner")

# ---------- identity ----------
with st.sidebar:
    st.header("Who are you?")
    learner_id = st.text_input("Learner ID", value=st.session_state.get("learner_id", "L001"))
    learner_name = st.text_input("Name", value=st.session_state.get("learner_name", "Niya"))
    st.session_state["learner_id"] = learner_id
    st.session_state["learner_name"] = learner_name
    st.caption("Use the same Learner ID across pages to keep your session.")

concepts = api.get_concepts()
concept_names = [c["name"] for c in concepts]

# ---------- Step 1-2: goal intake ----------
st.subheader("1. Tell the system your goal")
col1, col2 = st.columns([2, 1])
with col1:
    goal_text = st.text_input("What do you want to learn?", "Teach me Python for my DBMS elective")
with col2:
    known = st.multiselect("I already know:", concept_names)

if st.button("Submit goal / Update plan", type="primary"):
    with st.spinner("Learner Understanding Agent -> Curriculum & Planning Agent -> Verification..."):
        result = api.submit_goal(learner_id, learner_name, goal_text, known)
    st.session_state["last_goal_result"] = result

result = st.session_state.get("last_goal_result")

# ---------- pipeline trace (Section 10, step 4: show the agents working) ----------
if result:
    with st.expander("🔎 Agent pipeline trace", expanded=True):
        st.code(
            "Learner Understanding Agent: profile built from goal + known concepts\n"
            f"Curriculum & Planning Agent: ordered {len(result['concept_order'])} unmet concepts\n"
            f"Neuro-Symbolic Verification: {result['verification']['status'].upper()}"
            + (f" ({result['verification']['reason']})" if result['verification']['reason'] else ""),
            language="text",
        )

    if result["status"] == "pending_review":
        st.warning(
            "⚠️ This plan was **flagged** by the verification layer and sent to the "
            "Educator dashboard for review. It won't reach you until approved."
        )
        st.markdown(f"**XAI explanation:** {result['explanation']}")
        with st.expander("Proposed (unreviewed) concept order"):
            st.write([c for c in result["concept_order"]])
    else:
        st.success("✅ Plan verified against all curriculum rules.")
        st.markdown(f"**XAI explanation:** {result['explanation']}")

st.divider()

# ---------- Step 5: active plan + content ----------
st.subheader("2. Your active learning plan")
plan = api.get_active_plan(learner_id)

if not plan:
    st.info("No active plan yet — submit a goal above, or wait for educator approval if it was flagged.")
else:
    concept_map = {c["concept_id"]: c["name"] for c in concepts}
    st.write(" → ".join(concept_map.get(c, c) for c in plan["concept_order"]))
    st.caption(f"Verified: {plan['verification_status']} · plan #{plan['plan_id']}")

    current_concept = plan["concept_order"][0] if plan["concept_order"] else None

    if current_concept:
        st.subheader(f"3. Study: {concept_map.get(current_concept, current_concept)}")
        content = api.get_content(current_concept)
        st.markdown(f"📘 {content['explanation']}")
        st.caption(f"Resource: {content['resource']}")

        st.subheader("4. Quick check")
        st.write(content["question"])
        given = st.text_input("Your answer", key=f"answer_{current_concept}")
        if st.button("Submit answer"):
            eval_result = api.submit_answer(
                learner_id, current_concept, content["question"], content["answer"], given
            )
            if eval_result["new_mastery"] > eval_result["old_mastery"]:
                st.success(f"✅ Correct-ish! Mastery on this concept: {eval_result['new_mastery']:.0%}")
            else:
                st.error(f"❌ Not quite (correct answer: '{content['answer']}'). "
                         f"Mastery on this concept: {eval_result['new_mastery']:.0%}")
            if eval_result["is_gap"]:
                st.warning(f"📉 Learning gap flagged on **{concept_map.get(current_concept)}**.")

        st.subheader("5. Re-plan if gaps were found")
        if st.button("Check for gaps & re-plan"):
            with st.spinner("Progress Evaluation Agent -> re-planning if needed..."):
                rp = api.replan(learner_id)
            if not rp["replanned"]:
                st.info("No learning gaps detected — plan stays as-is.")
            else:
                st.markdown(f"**Gaps found:** {[concept_map.get(g, g) for g in rp['gaps']]}")
                st.markdown(f"**XAI explanation:** {rp['explanation']}")
                if rp["status"] == "pending_review":
                    st.warning("The re-plan was flagged and sent to the Educator for review.")
                else:
                    st.success("Re-plan verified and applied — refresh the plan above.")

st.divider()

# ---------- mastery overview ----------
st.subheader("Your mastery so far")
profile = api.get_profile(learner_id)
if profile and profile.get("mastery"):
    concept_map = {c["concept_id"]: c["name"] for c in concepts}
    for cid, score in profile["mastery"].items():
        st.progress(min(max(score, 0.0), 1.0), text=f"{concept_map.get(cid, cid)} — {score:.0%}")
else:
    st.caption("No mastery data yet.")
