# streamlit_app.py 
import streamlit as st 
import streamlit.components.v1 as components 
import pandas as pd 
import json 
from datetime import datetime, time as dt_time 
from io import StringIO 
from typing import List, Dict, Any, Set, Tuple 
import time as time_module 
from urllib.parse import quote_plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Streamlit App Layout ----------
st.title("✨ Aurora – Personalized Learning Assistant")
st.caption("Your personalized study planner with feedback, progress tracking, and topic recommendations.")

# Main navigation
tab_onboard, tab_paths, tab_timer = st.tabs(["Onboarding", "Learning Path", "Timer"])

# ------------------- ONBOARDING -------------------
with tab_onboard:
    st.subheader("Profile Setup")
    st.markdown("Tell Aurora about your learning preferences to create your personalized schedule.")

    with st.form("onboard_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(" Full Name", placeholder="e.g., Shine Sara Mathew")
            preferred_time = st.time_input("⏰ Preferred Study Time", value=dt_time(20, 30))
            subject = st.selectbox(
                "🎯 Interest / Subject Area",
                [
                    "Artificial Intelligence", "Web Development", "Biomaterials", "Medical Imaging",
                    "Bioinformatics Biotechnology", "Nanobiotechnology", "Renewable Energy in Agriculture",
                    "Geoinformatics and Remote Sensing in Agriculture", "Food Microbiology and Safety",
                    "Emerging Areas in Food Processing", "Other"
                ]
            )
            if subject == "Other":
                subject = st.text_input("Specify Subject Area")
        with col2:
            st.markdown("**Session Preferences**")
            hours = st.selectbox("⏳ Hours", [0, 1, 2, 3, 4, 5, 6], index=1)
            minutes = st.selectbox("⏱️ Minutes", [0, 15, 30, 45], index=0)
            duration_unit = st.selectbox("Duration Unit", ["Weeks", "Months"], index=0)
            duration_amount = st.number_input("Duration Amount", min_value=1, max_value=52, value=4, step=1)
            repeat_per_week = st.slider("Sessions per Week", 1, 14, 3)

        goals = st.text_area("💭 Learning Goals", placeholder="E.g., build a project, revise daily, pass exams...")
        submitted = st.form_submit_button("💾 Save Profile")

        if submitted:
            if not name:
                st.error("⚠️ Please enter your name")
            elif hours == 0 and minutes == 0:
                st.error("⚠️ Please set a session length greater than 0")
            else:
                st.session_state.profile = {
                    'name': name, 'preferred_time': preferred_time, 'subject': subject,
                    'hours': hours, 'minutes': minutes, 'duration_amount': duration_amount,
                    'duration_unit': duration_unit, 'sessions_per_week': repeat_per_week, 'goals': goals
                }
                st.success("✅ Profile saved successfully! Go to **Learning Path** to generate your plan.")

    if st.session_state.profile:
        with st.expander("Current Profile Summary"):
            st.json(st.session_state.profile, expanded=False)

# ------------------- LEARNING PATH -------------------
with tab_paths:
    st.subheader("Generate Your Learning Path")
    st.markdown("Aurora recommends study topics and resources based on your interests and skill level.")

    # Generate path controls
    left_col, right_col = st.columns([2, 1])
    with left_col:
        profile = st.session_state.get("profile")
        if profile:
            default_interest = profile.get("subject", "")
            target_weeks = profile['duration_amount'] if profile['duration_unit'] == "Weeks" else profile['duration_amount'] * 4
            hours_per_week = max(1.0, profile['hours'] + profile['minutes'] / 60.0) * profile['sessions_per_week']
        else:
            default_interest, target_weeks, hours_per_week = "", 4, 6.0

        interests_input = st.text_input("💡 Your Interests (comma separated)", value=default_interest)
        skill_level = st.slider("🧠 Skill Level", 1, 5, 3)
        interests = [i.strip() for i in interests_input.split(",") if i.strip()]
    with right_col:
        resource_types = st.multiselect("🎥 Resource Types", ["video", "article", "interactive", "other"], default=["video", "article", "interactive"])
        require_resources = st.checkbox("Require topics with chosen resource types", value=False)
        max_seed = st.number_input("Topic Breadth (Top N)", 1, 20, 8)
def ml_rank_topics(topics: List[Dict[str, Any]], user_interest_text: str) -> List[Tuple[str, float]]:
    """
    Compute semantic relevance scores using TF-IDF + cosine similarity.
    Returns a list of (topic_id, similarity_score) sorted descending.
    """
    corpus, topic_ids = [], []
    for t in topics:
        txt = f"{t.get('title','')} " + " ".join(t.get('tags', [])) + " " + \
              " ".join(r.get('title','') for r in t.get('resources', []))
        corpus.append(txt)
        topic_ids.append(t['id'])

    if not corpus:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    if not user_interest_text.strip():
        user_interest_text = "general learning"
    user_vec = vectorizer.transform([user_interest_text])

    sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
    ranked = sorted(zip(topic_ids, sims), key=lambda x: x[1], reverse=True)
    return ranked

        if st.button("🚀 Generate Learning Path"):
            topics = load_topics_from_df(_default_df)
            res = def generate_path(topics: List[Dict[str, Any]], interests: List[str], skill_level: int,
                  hours_per_week: float, max_seed=8, target_weeks: int = None,
                  resource_types: List[str] = None, require_resources: bool = False):

    by_id = {t['id']: t for t in topics}

    # Combine user interests into a single text string
    interest_text = " ".join(interests)
    ml_ranked = ml_rank_topics(topics, interest_text)
    ml_scores = {tid: score for tid, score in ml_ranked}

    # Combine ML similarity + difficulty penalty
    def score_topic(t):
        base = ml_scores.get(t['id'], 0.0) * 100
        diff_penalty = max(0, t.get('difficulty', 3) - skill_level) * 3
        return base - diff_penalty

    scored = []
    for t in topics:
        if require_resources and resource_types:
            rtypes = [r.get('type', '').lower() for r in t.get('resources', [])]
            if not any(rt in rtypes for rt in resource_types):
                continue
        s = score_topic(t)
        scored.append({**t, "score": s})

    # Sort by ML relevance
    scored.sort(key=lambda x: x['score'], reverse=True)

    # Pick seeds
    seed = []
    for t in scored:
        if len(seed) < max_seed:
            seed.append(t['id'])

    # Include prerequisites recursively
    needed: Set[str] = set()
    def include_rec(id_):
        if id_ in needed: return
        if id_ not in by_id: return
        needed.add(id_)
        for p in by_id[id_].get('prereqs', []):
            include_rec(p)
    for sid in seed:
        include_rec(sid)

    selected = [by_id[i] for i in needed if i in by_id]
    ordered_ids, cycle = topo_sort_with_prereqs(selected)
    ordered = [oid for oid in ordered_ids if oid in by_id and oid in needed]
    ordered_topics = [by_id[i] for i in ordered]

    # --- Weekly scheduling logic (keep same as before) ---
    weeks = []
    total_hours = sum(float(t.get('est_hours', 0)) for t in ordered_topics)
    if target_weeks and target_weeks > 0:
        per_week = total_hours / target_weeks if total_hours > 0 else max(1.0, hours_per_week)
        weeks = [{"hours_left": per_week, "topics": []} for _ in range(target_weeks)]
        week_idx = 0
        for t in ordered_topics:
            dur = float(t.get('est_hours', 2.0))
            remaining = dur
            while remaining > 0:
                if week_idx >= len(weeks):
                    weeks.append({"hours_left": 0.0, "topics": []})
                capacity = weeks[week_idx]['hours_left']
                if capacity <= 0:
                    week_idx += 1
                    continue
                use = min(remaining, capacity)
                note = "start" if remaining == dur and remaining > use else \
                       ("continue" if remaining > use else "finish")
                weeks[week_idx]['topics'].append({**t, "scheduled_hours": use, "note": note})
                weeks[week_idx]['hours_left'] -= use
                remaining -= use
                if weeks[week_idx]['hours_left'] <= 1e-6:
                    week_idx += 1
    else:
        current_week = {"hours_left": hours_per_week, "topics": []}
        for t in ordered_topics:
            dur = float(t.get('est_hours', 2.0))
            if dur <= current_week['hours_left']:
                current_week['topics'].append({**t, "scheduled_hours": dur})
                current_week['hours_left'] -= dur
            else:
                if current_week['topics']:
                    weeks.append(current_week)
                    current_week = {"hours_left": hours_per_week, "topics": []}
                remaining = dur
                first = True
                while remaining > 0:
                    use = min(remaining, current_week['hours_left'] if current_week['hours_left']>0 else hours_per_week)
                    note = "start" if first and remaining>use else \
                           ("continue" if remaining>use else "finish")
                    current_week['topics'].append({**t, "scheduled_hours": use, "note": note})
                    remaining -= use
                    current_week['hours_left'] -= use
                    first = False
                    if remaining > 0:
                        weeks.append(current_week)
                        current_week = {"hours_left": hours_per_week, "topics": []}
        if current_week['topics']:
            weeks.append(current_week)

    meta = {"generated_at": datetime.utcnow().isoformat() + "Z", "skill_level": skill_level,
            "hours_per_week": hours_per_week, "target_weeks": target_weeks,
            "resource_types": resource_types, "require_resources": require_resources}
    return {"ordered": ordered_topics, "weeks": weeks, "meta": meta, "cycle_detected": cycle}

            st.session_state.last_result = res
            st.success("🎯 Path generated successfully! Scroll down to explore your topics.")

    # Display generated path
    if st.session_state.last_result:
        res = st.session_state.last_result
        ordered, weeks = res["ordered"], res["weeks"]

        st.divider()
        st.markdown("### Recommended Topics (Ordered)")

        for i, t in enumerate(ordered, 1):
            with st.expander(f"{i}. {t['title']} ({t.get('est_hours', '?')} hrs)"):
                st.write("**Tags:**", ", ".join(t.get('tags', [])) or "_None_")
                st.write("**Prerequisites:**", ", ".join(t.get('prereqs', [])) or "_None_")

                for r_idx, r in enumerate(t.get("resources", [])):
                    url = r.get("url") or f"https://www.google.com/search?q={quote_plus(r.get('title',''))}"
                    colA, colB = st.columns([4, 1])
                    with colA:
                        st.markdown(f"- **{r.get('type','').title()}**: [{r.get('title','')}]({url})")
                    with colB:
                        if st.button("Open & Start", key=f"start_{i}_{r_idx}"):
                            start_session_with_url(url, t['title'], r.get('title'))

        st.divider()
        st.markdown("### 🗓️ Weekly Plan Overview")
        for idx, week in enumerate(weeks):
            st.markdown(f"**Week {idx+1}:**")
            if not week["topics"]:
                st.write("_No topics assigned this week._")
            for tt in week["topics"]:
                st.write(f"- {tt['title']} ({round(tt.get('scheduled_hours',0),1)} hrs)")

        st.download_button("📥 Download Path (JSON)", json.dumps(res, indent=2), "study_path.json")

# ------------------- TIMER -------------------
with tab_timer:
    profile = st.session_state.get("profile")
    st.subheader("⏱️ Study Session Timer")

    if not profile:
        st.info("Please complete onboarding and generate a path to start learning.")
    else:
        total_time = (profile["hours"] * 3600) + (profile["minutes"] * 60)
        if st.session_state.timer_running and st.session_state.start_time:
            elapsed = (datetime.now() - st.session_state.start_time).total_seconds()
            current_time_left = max(0, int(st.session_state.time_left - elapsed))
            if current_time_left <= 0:
                st.session_state.show_celebration = True
                st.session_state.timer_running = False
                st.balloons()
        else:
            current_time_left = st.session_state.time_left

        if not st.session_state.show_celebration:
            st.markdown(f"### ⏰ {format_time(current_time_left)} remaining")
            status = "🟢 Running" if st.session_state.timer_running else "⏸️ Paused"
            st.caption(status)
            progress = 1 - (current_time_left / total_time) if total_time > 0 else 0
            st.progress(progress)
            c1, c2, c3 = st.columns(3)
            with c2:
                if st.button("⏯ Pause / Resume"):
                    pause_timer()
                if st.button("🔁 Reset Timer"):
                    reset_timer()
        else:
            st.success("🎉 Session Completed!")
            st.write("How did you feel about your learning experience?")
            cols = st.columns(5)
            emojis = ["😃", "🙂", "😐", "😣", "😭"]
            labels = ["Loved it", "Good", "Okay", "Struggled", "Too hard"]
            for i, (e, l) in enumerate(zip(emojis, labels)):
                with cols[i]:
                    if st.button(f"{e} {l}", key=f"f_{i}"):
                        st.session_state.feedbacks.append({"feeling": l, "timestamp": datetime.utcnow().isoformat()})
                        st.success("Feedback recorded!")

# Auto-refresh
if st.session_state.timer_running:
    time_module.sleep(1)
    safe_rerun()
