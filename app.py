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

# --- Initialize all session state keys ---
if "profile" not in st.session_state:
    st.session_state.profile = None
if "timer_running" not in st.session_state:
    st.session_state.timer_running = False
if "time_left" not in st.session_state:
    st.session_state.time_left = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "paused_time" not in st.session_state:
    st.session_state.paused_time = 0
if "show_celebration" not in st.session_state:
    st.session_state.show_celebration = False
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ---------- Theme switcher (paste after session_state initialization) ----------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"  # default

# Theme selector in sidebar (or use st.selectbox somewhere in UI)
with st.sidebar:
    st.markdown("### 🎨 Theme")
    chosen = st.radio("", options=["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
    if chosen != st.session_state.theme:
        st.session_state.theme = chosen

# CSS definitions
LIGHT_CSS = """
<style>
.stApp { background: linear-gradient(160deg, #f9e7d1 0%, #d8b67a 100%); color: #3e2c23; font-family: 'Georgia', serif; }
div[data-testid="stForm"], .stExpander { background-color: rgba(255, 248, 240, 0.95); border-radius: 16px; padding: 20px; border: 1px solid #e2c59b; box-shadow: 0 4px 12px rgba(139,69,19,0.12); }
.stTextInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>div>textarea, .stTimeInput>div>div>input { background-color: #fffaf4; color: #3e2c23; border-radius: 10px; border: 1px solid #d6ad60; padding: 8px 12px; }
label, .stMarkdown p, .stSlider label { color: #4a2e12 !important; font-weight: 600 !important; }
.stButton>button { background-color: #a0522d !important; color: #fff !important; font-weight: 600; border-radius: 10px; padding: 10px 24px; }
.stButton>button:hover { background-color: #8b3c15 !important; }
.stTabs [data-baseweb="tab"] { background-color: #f5e3c5; color: #3e2c23; border-radius: 10px 10px 0 0; padding: 8px 14px; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #a0522d; color: white; }
h1,h2,h3 { color: #5c3317; }
</style>
"""

DARK_CSS = """
<style>
/* Dark theme */
.stApp { background: linear-gradient(180deg, #0f1720 0%, #111827 100%); color: #e6d8be; font-family: 'Georgia', serif; }
div[data-testid="stForm"], .stExpander { background-color: rgba(18, 23, 28, 0.85); border-radius: 12px; padding: 18px; border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 6px 24px rgba(2,6,23,0.6); }
.stTextInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>div>textarea, .stTimeInput>div>div>input { background-color: #0b1014; color: #e6d8be; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04); padding: 8px 12px; }
label, .stMarkdown p, .stSlider label { color: #e6d8be !important; font-weight: 600 !important; }
.stButton>button { background-color: #f59e0b !important; color: #08121a !important; font-weight: 700; border-radius: 8px; padding: 8px 20px; }
.stButton>button:hover { filter: brightness(0.95); }
.stTabs [data-baseweb="tab"] { background-color: rgba(255,255,255,0.02); color: #e6d8be; border-radius: 8px 8px 0 0; padding: 8px 14px; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #f59e0b; color: #08121a; }
h1,h2,h3 { color: #ffdca8; }
</style>
"""

# Apply selected CSS
if st.session_state.theme == "Dark":
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

# ---------- Streamlit App Layout ----------
st.title("✨ Aurora – Personalized Learning Assistant")
st.caption("Your personalized study planner with feedback, progress tracking, and topic recommendations.")

# ----------------- Helper functions (PASTE THIS BEFORE YOUR UI / tabs) -----------------

from io import StringIO
from datetime import datetime

def format_time(seconds: int) -> str:
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def parse_tags(cell) -> List[str]:
    if pd.isna(cell) or cell is None:
        return []
    return [t.strip() for t in str(cell).split(';') if t.strip()]

def parse_prereqs(cell) -> List[str]:
    if pd.isna(cell) or cell is None:
        return []
    return [s.strip() for s in str(cell).split(';') if s.strip()]

def parse_resources(cell) -> List[Dict[str,str]]:
    """
    Accepts either the pipe format like "video:Title|url:http..." or a JSON string.
    Returns list of {type, title, url?}.
    """
    if pd.isna(cell) or cell is None:
        return []
    txt = str(cell).strip()
    # try JSON first
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    items = []
    # split by pipe, collect resource entries and optional url keys
    parts = [p.strip() for p in txt.split('|') if p.strip()]
    for part in parts:
        if ':' in part:
            k, v = part.split(':', 1)
            k = k.strip().lower()
            v = v.strip()
            if k == 'url':
                if items:
                    items[-1]['url'] = v
                else:
                    items.append({'type':'link','title':v,'url':v})
            else:
                items.append({'type': k, 'title': v})
        else:
            items.append({'type': 'other', 'title': part})
    return items

def load_topics_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame into the internal topic dict list:
    Each topic: {id, title, tags, prereqs, difficulty, est_hours, resources}
    """
    topics = []
    for i, row in df.iterrows():
        tid = str(row.get("id")) if not pd.isna(row.get("id")) else f"r{i}"
        title = str(row.get("title")) if not pd.isna(row.get("title")) else tid
        tags = parse_tags(row.get("tags")) if "tags" in df.columns else []
        prereqs = parse_prereqs(row.get("prereqs")) if "prereqs" in df.columns else []
        try:
            difficulty = int(row.get("difficulty")) if not pd.isna(row.get("difficulty")) else 3
        except Exception:
            difficulty = 3
        try:
            est_hours = float(row.get("est_hours")) if not pd.isna(row.get("est_hours")) else 2.0
        except Exception:
            est_hours = 2.0
        resources = parse_resources(row.get("resources")) if "resources" in df.columns else []
        topics.append({
            "id": tid,
            "title": title,
            "tags": tags,
            "prereqs": prereqs,
            "difficulty": difficulty,
            "est_hours": est_hours,
            "resources": resources
        })
    return topics

def topo_sort_with_prereqs(topics: List[Dict[str,Any]]) -> Tuple[List[str], bool]:
    """Return topo order of ids, and whether a cycle was detected."""
    adj = {t['id']: list(t.get('prereqs', [])) for t in topics}
    visiting = {}
    order = []
    cycle = False
    def dfs(u):
        nonlocal cycle
        if visiting.get(u,0) == 1:
            cycle = True
            return
        if visiting.get(u,0) == 2:
            return
        visiting[u] = 1
        for v in adj.get(u, []):
            if v in adj:
                dfs(v)
        visiting[u] = 2
        order.append(u)
    for node in adj:
        if visiting.get(node,0) == 0:
            dfs(node)
    return list(reversed(order)), cycle

def generate_path(topics: List[Dict[str,Any]], interests: List[str], skill_level: int,
                  hours_per_week: float, max_seed=8, target_weeks: int = None,
                  resource_types: List[str] = None, require_resources: bool = False) -> Dict[str,Any]:
    """
    Lightweight path generator:
    - Scores topics by tag overlap with interests and difficulty penalty
    - Picks top-N seeds, includes prerequisites, topologically orders, schedules across target_weeks
    """
    by_id = {t['id']: t for t in topics}

    # basic scoring: tag overlap * weight - difficulty penalty
    interest_set = set([tok.lower() for i in interests for tok in i.split() if tok.strip()])
    def score_topic(t):
        t_tags = set([x.lower() for x in t.get('tags',[])])
        tag_score = len(t_tags & interest_set)
        diff_penalty = max(0, t.get('difficulty',3) - skill_level)
        return tag_score * 10 - diff_penalty * 3

    scored = []
    for t in topics:
        if require_resources and resource_types:
            rtypes = [r.get('type','').lower() for r in t.get('resources',[])]
            if not any(rt in rtypes for rt in resource_types):
                continue
        scored.append({**t, "score": score_topic(t)})

    scored.sort(key=lambda x: x['score'], reverse=True)
    seed = [t['id'] for t in scored[:max_seed]]

    # include prerequisites recursively
    needed = set()
    def include_rec(id_):
        if id_ in needed: return
        if id_ not in by_id: return
        needed.add(id_)
        for p in by_id[id_].get('prereqs', []):
            include_rec(p)
    for s in seed:
        include_rec(s)

    selected = [by_id[i] for i in needed if i in by_id]
    ordered_ids, cycle = topo_sort_with_prereqs(selected)
    ordered_topics = [by_id[i] for i in ordered_ids if i in by_id and i in needed]

    # schedule across weeks
    weeks = []
    total_hours = sum(float(t.get('est_hours',2.0)) for t in ordered_topics)
    if target_weeks and target_weeks > 0:
        per_week = total_hours / target_weeks if total_hours>0 else max(1.0, hours_per_week)
        weeks = [{"hours_left": per_week, "topics": []} for _ in range(target_weeks)]
        week_idx = 0
        for t in ordered_topics:
            dur = float(t.get('est_hours',2.0))
            remaining = dur
            while remaining > 0:
                if week_idx >= len(weeks):
                    weeks.append({"hours_left": per_week, "topics": []})
                capacity = weeks[week_idx]['hours_left']
                if capacity <= 0:
                    week_idx += 1
                    continue
                use = min(remaining, capacity)
                note = "start" if remaining == dur and remaining > use else ("continue" if remaining > use else "finish")
                weeks[week_idx]['topics'].append({**t, "scheduled_hours": use, "note": note})
                weeks[week_idx]['hours_left'] -= use
                remaining -= use
                if weeks[week_idx]['hours_left'] <= 1e-6:
                    week_idx += 1
    else:
        # simple packing into weeks based on hours_per_week
        current_week = {"hours_left": hours_per_week, "topics": []}
        for t in ordered_topics:
            dur = float(t.get('est_hours',2.0))
            if dur <= current_week['hours_left']:
                current_week['topics'].append({**t, "scheduled_hours": dur})
                current_week['hours_left'] -= dur
            else:
                if current_week['topics']:
                    weeks.append(current_week)
                remaining = dur
                while remaining > 0:
                    cap = current_week['hours_left'] if current_week['hours_left']>0 else hours_per_week
                    use = min(remaining, cap)
                    current_week['topics'].append({**t, "scheduled_hours": use})
                    current_week['hours_left'] -= use
                    remaining -= use
                    if current_week['hours_left'] <= 1e-6:
                        weeks.append(current_week)
                        current_week = {"hours_left": hours_per_week, "topics": []}
        if current_week['topics']:
            weeks.append(current_week)

    meta = {"generated_at": datetime.utcnow().isoformat() + "Z", "skill_level": skill_level,
            "hours_per_week": hours_per_week, "target_weeks": target_weeks}
    return {"ordered": ordered_topics, "weeks": weeks, "meta": meta, "cycle_detected": cycle}


# Simple session start helper (opens url and starts timer)
def start_session_with_url(url: str, topic_title: str = None, resource_title: str = None):
    profile = st.session_state.get("profile")
    if not profile:
        st.error("No profile set. Please create one in Onboarding.")
        return
    # record current context
    st.session_state.current_topic = topic_title
    st.session_state.current_resource = resource_title
    total_seconds = int((profile.get('hours',0) * 3600) + (profile.get('minutes',0) * 60))
    if total_seconds <= 0:
        total_seconds = 30*60
    st.session_state.time_left = total_seconds
    st.session_state.start_time = datetime.utcnow()
    st.session_state.timer_running = True
    st.session_state.show_celebration = False
    # open new tab
    safe_url = url.replace("'", "\\'")
    html = f"<script>window.open('{safe_url}', '_blank');</script>"
    components.html(html, height=0)
    # trigger rerun if available
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.rerun()
    except Exception:
        pass


def pause_timer():
    if st.session_state.timer_running:
        st.session_state.timer_running = False
        elapsed = (datetime.utcnow() - st.session_state.get("start_time", datetime.utcnow())).total_seconds() if st.session_state.get("start_time") else 0
        st.session_state.paused_time = max(0, st.session_state.get("time_left",0) - int(elapsed))
    else:
        st.session_state.timer_running = True
        st.session_state.start_time = datetime.utcnow()
        if st.session_state.get("paused_time",0) > 0:
            st.session_state.time_left = st.session_state.paused_time

def reset_timer():
    profile = st.session_state.get("profile")
    if not profile:
        return
    total_seconds = int((profile.get('hours',0) * 3600) + (profile.get('minutes',0) * 60))
    if total_seconds <= 0:
        total_seconds = 30*60
    st.session_state.time_left = total_seconds
    st.session_state.start_time = datetime.utcnow()
    st.session_state.timer_running = True
    st.session_state.show_celebration = False
    st.session_state.paused_time = 0

# ---------------------------------------------------------------------------------------

# Main navigation
tab_onboard, tab_paths, tab_timer = st.tabs(["🧭 Onboarding", "📚 Learning Path", "⏱️ Timer"])

# ------------------- ONBOARDING -------------------
with tab_onboard:
    st.subheader("Profile Setup")
    st.markdown("Tell Aurora about your learning preferences to create your personalized schedule.")

    with st.form("onboard_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("👤 Full Name", placeholder="Your name")
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
                subject = st.text_input("✏️ Specify Subject Area")
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
        with st.expander("📋 Current Profile Summary"):
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

        if st.button("🚀 Generate Learning Path"):
            topics = load_topics_from_df(_default_df)
            res = generate_path(
                topics, interests, skill_level, hours_per_week,
                max_seed, target_weeks, [r.lower() for r in resource_types], require_resources
            )
            st.session_state.last_result = res
            st.success("🎯 Path generated successfully! Scroll down to explore your topics.")

    # Display generated path
    if st.session_state.last_result:
        res = st.session_state.last_result
        ordered, weeks = res["ordered"], res["weeks"]

        st.divider()
        st.markdown("### 🧩 Recommended Topics (Ordered)")

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
