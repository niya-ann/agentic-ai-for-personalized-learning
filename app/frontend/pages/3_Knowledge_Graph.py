"""
pages/3_Knowledge_Graph.py — renders the KG for the demo (Section 10 step 3).
"""
import os
import streamlit as st
from app import kg as kg_mod

st.set_page_config(page_title="Knowledge Graph", page_icon="🕸️", layout="wide")
st.title("🕸️ Knowledge Graph")

g = kg_mod.load_graph()

st.markdown(f"**{g.number_of_nodes()} nodes, {g.number_of_edges()} edges** — "
            f"1 course, {len(kg_mod.get_concepts_for_course(g, kg_mod.SEED_COURSE))} concepts, "
            f"{len(kg_mod.get_rules_for_course(g, kg_mod.SEED_COURSE))} curriculum rules.")

learner_id = st.text_input("Highlight known concepts for learner", value=st.session_state.get("learner_id", "L001"))

known = []
try:
    from app import db
    profile = db.get_profile(learner_id)
    if profile:
        known = [c for c, s in profile["mastery"].items() if s >= 0.7]
except Exception:
    pass

img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "db_files", "kg_render.png")
kg_mod.render_png(g, img_path, highlight_known=known)
st.image(img_path, caption="Green = mastered by this learner (>=70%)")

st.subheader("Curriculum rules")
for rule in kg_mod.get_rules_for_course(g, kg_mod.SEED_COURSE):
    st.markdown(f"- **{rule['rule_id']}** ({rule['rule_kind']}): {rule['text']}")
