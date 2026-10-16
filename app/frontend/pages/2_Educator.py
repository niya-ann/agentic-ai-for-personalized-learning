"""
pages/2_Educator.py — the Human-in-the-Loop dashboard (Section 10 steps 12-13,
Section 15).

Any recommendation the verification layer flags lands here. The educator sees
the same XAI explanation and verification reason as the learner would have,
and can Approve / Modify / Reject before it ever reaches the learner.
"""
import streamlit as st
from app.frontend import api_client as api

st.set_page_config(page_title="Educator", page_icon="🧑‍🏫", layout="wide")
st.title("🧑‍🏫 Educator Dashboard")

concepts = api.get_concepts()
concept_map = {c["concept_id"]: c["name"] for c in concepts}

st.subheader("Pending recommendations")
pending = api.hitl_pending()

if not pending:
    st.info("No flagged recommendations waiting for review. 🎉")
else:
    for item in pending:
        proposal = item["proposal_json"]
        order = proposal.get("concept_order", [])
        with st.container(border=True):
            st.markdown(f"**Learner:** `{item['learner_id']}` · **Decision #{item['decision_id']}**")
            st.markdown(f"**Proposed plan:** " + " → ".join(concept_map.get(c, c) for c in order))
            st.markdown(f"**Verification reason:** {item['verification_reason']}")
            st.markdown(f"**XAI explanation:** {item['explanation']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Approve", key=f"approve_{item['decision_id']}"):
                    api.hitl_decide(item["decision_id"], "approved")
                    st.rerun()
            with col2:
                with st.popover("✏️ Modify"):
                    st.caption("Reorder by editing the concept list (comma-separated concept_ids).")
                    edited = st.text_input(
                        "Edited order", value=",".join(order), key=f"edit_{item['decision_id']}"
                    )
                    note = st.text_input("Note", key=f"note_{item['decision_id']}")
                    if st.button("Save modification", key=f"save_{item['decision_id']}"):
                        new_order = [c.strip() for c in edited.split(",") if c.strip()]
                        api.hitl_decide(item["decision_id"], "modified", new_order, note)
                        st.rerun()
            with col3:
                with st.popover("❌ Reject"):
                    reason = st.text_input("Reason for rejection", key=f"reject_reason_{item['decision_id']}")
                    if st.button("Confirm reject", key=f"reject_{item['decision_id']}"):
                        api.hitl_decide(item["decision_id"], "rejected", note=reason)
                        st.rerun()

st.divider()
st.subheader("Decision history")
history = api.hitl_history()
if not history:
    st.caption("No decisions made yet.")
else:
    for item in history:
        icon = {"approved": "✅", "modified": "✏️", "rejected": "❌"}.get(item["decision"], "•")
        st.write(f"{icon} `{item['learner_id']}` — {item['decision']} at {item['decided_at']}"
                 + (f" — note: {item['educator_note']}" if item.get("educator_note") else ""))
