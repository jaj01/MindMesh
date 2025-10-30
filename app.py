
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


# ---------- Utility Functions ----------
def safe_rerun():
    """Safe rerun compatible across Streamlit versions."""
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass
    try:
        if hasattr(st, "rerun"):
            st.rerun()
            return
    except Exception:
        pass


def format_time(seconds: int) -> str:
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


# ---------- Sample Dataset ----------
SAMPLE_CSV = """id,title,tags,prereqs,difficulty,est_hours,resources
t1,Computer Science Fundamentals,cs;fundamentals,,1,4,"article:CS Overview|url:https://en.wikipedia.org/wiki/Computer_science"
t2,Python Basics,python;programming,t1,1,6,"video:Python Crash Course|url:https://www.python.org"
t3,Data Structures,ds;algorithms,t1,3,8,"article:Intro to DS"
t4,Algorithms: Sorting & Searching,algorithms;ds,t3,3,6,"video:Sorting Algorithms"
t5,Web Development Basics,web;html;css,t1,2,6,"interactive:Build a page"
t6,Machine Learning Intro,ml;data,t2;t3,4,10,"video:ML Intro"
t7,SQL Basics,db;sql,t1,2,4,"article:SQL Tutorial"
t8,Project: Build a ToDo App,project;web,t5,2,8,"interactive:Project Guide"
"""
_default_df = pd.read_csv(StringIO(SAMPLE_CSV))


# ---------- Parsing Helpers ----------
def parse_tags(cell):
    if pd.isna(cell): return []
    return [t.strip() for t in str(cell).split(';') if t.strip()]


def parse_prereqs(cell):
    if pd.isna(cell): return []
    return [s.strip() for s in str(cell).split(';') if s.strip()]


def parse_resources(cell):
    if pd.isna(cell): return []
    txt = str(cell).strip()
    items = []
    for part in txt.split('|'):
        if ':' in part:
            k, v = part.split(':', 1)
            k, v = k.strip().lower(), v.strip()
            if k == 'url':
                if items:
                    items[-1]['url'] = v
                else:
                    items.append({'type': 'link', 'title': v, 'url': v})
            else:
                items.append({'type': k, 'title': v})
    return items


def load_topics_from_df(df):
    topics = []
    for _, row in df.iterrows():
        t = {
            "id": row.get("id", ""),
            "title": row.get("title", ""),
            "tags": parse_tags(row.get("tags", "")),
            "prereqs": parse_prereqs(row.get("prereqs", "")),
            "difficulty": int(row.get("difficulty", 3)),
            "est_hours": float(row.get("est_hours", 2.0)),
            "resources": parse_resources(row.get("resources", ""))
        }
        topics.append(t)
    return topics


# ---------- ML Topic Ranking ----------
def ml_rank_topics(topics: List[Dict[str, Any]], user_interest_text: str) -> List[Tuple[str, float]]:
    corpus, topic_ids = [], []
    for t in topics:
        txt = f"{t.get('title', '')} " + " ".join(t.get('tags', [])) + " " + \
              " ".join(r.get('title', '') for r in t.get('resources', []))
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


# ---------- Path Generator ----------
def topo_sort_with_prereqs(topics: List[Dict[str, Any]]) -> Tuple[List[str], bool]:
    adj = {t['id']: t.get('prereqs', []) for t in topics}
    visiting, order, cycle = {}, [], False

    def dfs(u):
        nonlocal cycle
        if visiting.get(u, 0) == 1:
            cycle = True
            return
        if visiting.get(u, 0) == 2:
            return
        visiting[u] = 1
        for v in adj.get(u, []):
            if v in adj:
                dfs(v)
        visiting[u] = 2
        order.append(u)

    for node in adj:
        if visiting.get(node, 0) == 0:
            dfs(node)
    return list(reversed(order)), cycle


def generate_path(topics, interests, skill_level, hours_per_week,
                  max_seed=8, target_weeks=None, resource_types=None, require_resources=False):

    by_id = {t['id']: t for t in topics}
    interest_text = " ".join(interests)
    ml_ranked = ml_rank_topics(topics, interest_text)
    ml_scores = {tid: score for tid, score in ml_ranked}

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
        scored.append({**t, "score": score_topic(t)})

    scored.sort(key=lambda x: x['score'], reverse=True)
    seed = [t['id'] for t in scored[:max_seed]]

    needed = set()
    def include_rec(i):
        if i in needed or i not in by_id: return
        needed.add(i)
        for p in by_id[i].get('prereqs', []):
            include_rec(p)
    for sid in seed:
        include_rec(sid)

    selected = [by_id[i] for i in needed]
    ordered_ids, cycle = topo_sort_with_prereqs(selected)
    ordered_topics = [by_id[i] for i in ordered_ids]

    # schedule by weeks
    total_hours = sum(t.get('est_hours', 2.0) for t in ordered_topics)
    weeks = []
    if target_weeks:
        per_week = total_hours / target_weeks
        weeks = [{"hours_left": per_week, "topics": []} for _ in range(target_weeks)]
        wi = 0
        for t in ordered_topics:
            dur = t.get('est_hours', 2.0)
            while dur > 0:
                if wi >= len(weeks):
                    weeks.append({"hours_left": per_week, "topics": []})
                use = min(weeks[wi]['hours_left'], dur)
                weeks[wi]['topics'].append({**t, "scheduled_hours": use})
                weeks[wi]['hours_left'] -= use
                dur -= use
                if weeks[wi]['hours_left'] <= 0: wi += 1
    return {"ordered": ordered_topics, "weeks": weeks, "meta": {"cycle": cycle}}


# ---------- Streamlit App ----------
st.title("✨ Aurora – Personalized Learning Assistant")
st.caption("Your personalized study planner with feedback, progress tracking, and ML-based topic recommendations.")

tab_onboard, tab_paths, tab_timer = st.tabs(["Onboarding", "Learning Path", "Timer"])


# ---------- Onboarding ----------
with tab_onboard:
    st.header("Profile Setup")
    with st.form("onboard_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Full Name")
            preferred_time = st.time_input("Preferred Study Time", value=dt_time(20, 30))
            subject = st.text_input("Interest / Subject Area", placeholder="e.g., Machine Learning, Python")
        with c2:
            hours = st.selectbox("Session Hours", [0, 1, 2, 3, 4], index=1)
            minutes = st.selectbox("Session Minutes", [0, 15, 30, 45], index=1)
            duration_amount = st.number_input("Duration (weeks)", min_value=1, value=4)
            repeat_per_week = st.slider("Sessions per week", 1, 14, 3)
        goals = st.text_area("Learning Goals")
        if st.form_submit_button("Save Profile"):
            st.session_state.profile = {
                "name": name, "preferred_time": preferred_time, "subject": subject,
                "hours": hours, "minutes": minutes, "duration_amount": duration_amount,
                "sessions_per_week": repeat_per_week, "goals": goals
            }
            st.success("Profile saved!")

    if st.session_state.get("profile"):
        st.json(st.session_state.profile)


# ---------- Learning Path ----------
with tab_paths:
    st.header("Generate Personalized Path")
    profile = st.session_state.get("profile")
    if profile:
        interests = [x.strip() for x in profile['subject'].split(',')]
        skill_level = st.slider("Skill Level", 1, 5, 3)
        resource_types = st.multiselect("Resource Types", ["video", "article", "interactive"], ["video", "article"])
        if st.button("Generate Path"):
            topics = load_topics_from_df(_default_df)
            res = generate_path(
                topics, interests, skill_level,
                hours_per_week=profile['hours'] * profile['sessions_per_week'],
                max_seed=8, target_weeks=profile['duration_amount'],
                resource_types=resource_types
            )
            st.session_state.last_result = res
            st.success("Path generated successfully!")

    if st.session_state.get("last_result"):
        res = st.session_state.last_result
        for i, t in enumerate(res["ordered"], 1):
            with st.expander(f"{i}. {t['title']}"):
                st.write("Tags:", ", ".join(t['tags']))
                for r in t.get("resources", []):
                    url = r.get("url") or f"https://www.google.com/search?q={quote_plus(r.get('title',''))}"
                    st.markdown(f"- **{r.get('type','').title()}**: [{r.get('title','')}]({url})")


# ---------- Timer ----------
with tab_timer:
    st.header("Study Timer")
    st.info("Timer functionality placeholder (from your previous code).")
