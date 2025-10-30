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

# Page config
st.set_page_config(page_title="Aurora - Personalized Learning", layout="wide", page_icon="✏️")

# ---------- safe rerun helper ----------
def safe_rerun():
    """Try the available Streamlit rerun API; show a friendly message if none exists."""
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
    st.warning("Automatic refresh not supported by this Streamlit version — please refresh manually if UI doesn't update.")

# ---------- CSS ----------
st.markdown("""
<style>
html, body, [class^="css"] { font-family: 'Georgia', serif; }
.stApp { background: linear-gradient(160deg, #f5f0e6 0%, #7a3515 100%); color: #3e2c23; padding: 0; }
.card { background: rgba(210,180,140,0.3); border-radius: 20px; padding: 24px; box-shadow: 0 10px 40px rgba(160,82,45,0.4); border: 1px solid rgba(160,82,45,0.5); margin-bottom: 20px; }
h1,h2,h3 { color: #8b4513; font-weight: bold; }
.muted { color:#a0522d; }
button, .stButton>button { background: #a0522d !important; color: #fff !important; font-weight:600; border-radius:12px; padding:8px 16px; }
.timer-display { font-size:3.2rem; font-weight:bold; text-align:center; color:#8b4513; margin:1.2rem 0; text-shadow:2px 2px 4px rgba(0,0,0,0.1); }
.celebration { text-align:center; padding:2rem; animation:bounce 1s infinite; }
@keyframes bounce { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-20px);} }
.stat-card { background: rgba(139,69,19,0.15); border-radius:15px; padding:20px; text-align:center; border:2px solid rgba(160,82,45,0.3); }
.small-muted { color:#8b451d; font-size:0.9rem; }
.resource-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
.emoji-btn { font-size:1.6rem; padding:6px 10px; border-radius:8px; border:1px solid rgba(0,0,0,0.06); background: #fff; cursor:pointer; }
.selected-emoji { box-shadow: 0 4px 12px rgba(0,0,0,0.08); background:#ffe9d6; }
</style>
""", unsafe_allow_html=True)

# ---------- Session state defaults ----------
if 'profile' not in st.session_state: st.session_state.profile = None
if 'timer_running' not in st.session_state: st.session_state.timer_running = False
if 'time_left' not in st.session_state: st.session_state.time_left = 0
if 'start_time' not in st.session_state: st.session_state.start_time = None
if 'paused_time' not in st.session_state: st.session_state.paused_time = 0
if 'show_celebration' not in st.session_state: st.session_state.show_celebration = False
if 'last_result' not in st.session_state: st.session_state.last_result = None
if 'cleaned_df' not in st.session_state: st.session_state.cleaned_df = None
if 'data_issues' not in st.session_state: st.session_state.data_issues = []
if 'current_topic' not in st.session_state: st.session_state.current_topic = None
if 'current_resource' not in st.session_state: st.session_state.current_resource = None
if 'feedbacks' not in st.session_state: st.session_state.feedbacks = []  # list of feedback dicts

# ---------- Embedded backend dataset ----------
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

# ---------- Helpers ----------
def format_time(seconds: int) -> str:
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def parse_tags(cell: Any) -> List[str]:
    if pd.isna(cell): return []
    return [t.strip() for t in str(cell).split(';') if t.strip()]

def parse_prereqs(cell: Any) -> List[str]:
    if pd.isna(cell): return []
    return [s.strip() for s in str(cell).split(';') if s.strip()]

def parse_resources(cell: Any) -> List[Dict[str,str]]:
    if pd.isna(cell): return []
    txt = str(cell).strip()
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, list): 
            return parsed
    except Exception:
        pass
    items = []
    for part in txt.split('|'):
        part = part.strip()
        if not part: continue
        if ':' in part:
            key, val = part.split(':', 1)
            if key.strip().lower() == 'url':
                if items:
                    items[-1]['url'] = val.strip()
                else:
                    items.append({"type":"link","title":val.strip(), "url":val.strip()})
            else:
                items.append({"type": key.strip().lower(), "title": val.strip()})
        else:
            items.append({"type":"other", "title": part})
    return items

def load_topics_from_df(df: pd.DataFrame) -> List[Dict[str,Any]]:
    topics = []
    for i, row in df.iterrows():
        try:
            diff = int(row.get("difficulty")) if not pd.isna(row.get("difficulty")) else 3
        except Exception:
            diff = 3
        try:
            est = float(row.get("est_hours")) if not pd.isna(row.get("est_hours")) else 2.0
        except Exception:
            est = 2.0
        tid = str(row.get("id")).strip() if not pd.isna(row.get("id")) else f"r{i}"
        t = {
            "id": tid,
            "title": str(row.get("title")) if not pd.isna(row.get("title")) else tid,
            "tags": parse_tags(row.get("tags")) if "tags" in df.columns else [],
            "prereqs": parse_prereqs(row.get("prereqs")) if "prereqs" in df.columns else [],
            "difficulty": diff,
            "est_hours": est,
            "resources": parse_resources(row.get("resources")) if "resources" in df.columns else []
        }
        topics.append(t)
    return topics

def data_cleaning_report(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    issues = []
    cleaned = df.copy()
    for col in ("est_hours", "difficulty"):
        if col in cleaned.columns:
            coerced = pd.to_numeric(cleaned[col], errors='coerce')
            bad_rows = cleaned[coerced.isna()].index.tolist()
            for r in bad_rows:
                issues.append(f"Row {r}: '{col}' invalid -> set to default")
            cleaned[col] = coerced.fillna(2 if col=="est_hours" else 3)
    if "resources" in cleaned.columns:
        empty_res = cleaned[cleaned["resources"].isna() | (cleaned["resources"].astype(str).str.strip()=="")].index.tolist()
        for r in empty_res:
            issues.append(f"Row {r}: resources empty (no resources listed)")
    return cleaned, issues

def topo_sort_with_prereqs(topics: List[Dict[str,Any]]) -> Tuple[List[str], bool]:
    adj = {}
    for t in topics:
        adj[t['id']] = [p for p in (t.get('prereqs') or [])]
    visiting = {}
    order = []
    cycle = False
    def dfs(u):
        nonlocal cycle
        state = visiting.get(u, 0)
        if state == 1:
            cycle = True
            return
        if state == 2:
            return
        visiting[u] = 1
        for v in adj.get(u, []):
            if v not in adj:
                continue
            dfs(v)
        visiting[u] = 2
        order.append(u)
    for node in list(adj.keys()):
        if visiting.get(node, 0) == 0:
            dfs(node)
    return list(reversed(order)), cycle

def generate_path(topics: List[Dict[str,Any]], interests: List[str], skill_level: int,
                  hours_per_week: float, max_seed=8, target_weeks: int = None,
                  resource_types: List[str] = None, require_resources: bool = False):
    by_id = {t['id']: t for t in topics}
    def score_topic(t):
        tag_score = sum(1 for tag in t['tags'] if tag in interests)
        diff_penalty = max(0, t.get('difficulty',3) - skill_level)
        pop = t.get('popularity', 0)
        return tag_score * 10 - diff_penalty * 3 + pop
    scored = []
    for t in topics:
        if require_resources and resource_types:
            rtypes = [r.get('type','').lower() for r in t.get('resources',[])]
            if not any(rt in rtypes for rt in resource_types):
                continue
        s = score_topic(t)
        scored.append({**t, "score": s})
    scored.sort(key=lambda x: x['score'], reverse=True)
    seed = []
    for t in scored:
        if len(seed) < max_seed and (t['score'] >= 0 or any(tag in interests for tag in t['tags'])):
            seed.append(t['id'])
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
    if cycle:
        selected_sorted = sorted(selected, key=lambda x: -x.get('score',0))
        ordered = [t['id'] for t in selected_sorted]
    else:
        ordered = [oid for oid in ordered_ids if oid in by_id and oid in needed]
    ordered_topics = [by_id[i] for i in ordered]

    # schedule across target_weeks if provided (force span)
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
                note = "start" if remaining == dur and remaining > use else ("continue" if remaining > use else "finish")
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
                    note = "start" if first and remaining>use else ("continue" if remaining>use else "finish")
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

# ---------- Timer helpers ----------
def start_session_with_url(url: str, topic_title: str = None, resource_title: str = None):
    """Open resource url in new tab and start the session timer (from profile session length).
       Also records current topic & resource in session_state for feedback later.
    """
    profile = st.session_state.profile
    if not profile:
        st.error("No profile found — set up profile in Onboarding first.")
        return
    # record current context for feedback
    st.session_state.current_topic = topic_title
    st.session_state.current_resource = resource_title
    # compute session duration from profile
    total_seconds = int((profile['hours'] * 3600) + (profile['minutes'] * 60))
    if total_seconds <= 0:
        total_seconds = 30 * 60  # fallback 30 minutes
    st.session_state.time_left = total_seconds
    st.session_state.start_time = datetime.now()
    st.session_state.timer_running = True
    st.session_state.show_celebration = False
    # open URL in new tab using components.html
    safe_url = url.replace("'", "\\'")
    html = f"<script>window.open('{safe_url}', '_blank');</script>"
    components.html(html, height=0)
    safe_rerun()

def pause_timer():
    if st.session_state.timer_running:
        st.session_state.timer_running = False
        elapsed = (datetime.now() - st.session_state.start_time).total_seconds() if st.session_state.start_time else 0
        st.session_state.paused_time = max(0, st.session_state.time_left - int(elapsed))
    else:
        st.session_state.timer_running = True
        st.session_state.start_time = datetime.now()
        if st.session_state.paused_time > 0:
            st.session_state.time_left = st.session_state.paused_time

def reset_timer():
    profile = st.session_state.profile
    if not profile:
        return
    total_seconds = int((profile['hours'] * 3600) + (profile['minutes'] * 60))
    if total_seconds <= 0:
        total_seconds = 30*60
    st.session_state.time_left = total_seconds
    st.session_state.start_time = datetime.now()
    st.session_state.timer_running = True
    st.session_state.show_celebration = False
    st.session_state.paused_time = 0

# ---------- Tabs: Onboarding | Timer | Paths ----------
tab_onboard, tab_timer, tab_paths = st.tabs(["Onboarding", "Timer", "Paths"])

# ---------- Onboarding ----------
with tab_onboard:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1>Aurora - Personalized Learning ✏️</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Setup your profile — this controls timer & path weeks</div>", unsafe_allow_html=True)
    with st.form("onboard_form"):
        c1, c2 = st.columns([1,1])
        with c1:
            name = st.text_input("👤 Full name", placeholder="e.g. Shine Sara Mathew")
            preferred_time = st.time_input("⏰ Preferred study time", value=dt_time(20,30))
            subject = st.selectbox("📚 Interest / Subject area",
                                   ["Artificial Intelligence","Web Development","Biomaterials","Medical Imaging",
                                    "Bioinformatics Biotechnology","Nanobiotechnology","Renewable Energy in Agriculture",
                                    "Geoinformatics and Remote Sensing in Agriculture","Food Microbiology and Safety",
                                    "Emerging Areas in Food Processing","Other"])
            if subject == "Other":
                subject = st.text_input("✏️ Please specify subject area", value="")
        with c2:
            st.markdown("🍂 **Preferred session length**")
            hours = st.selectbox("⏳ Hours", options=[0,1,2,3,4,5,6], index=0)
            minutes = st.selectbox("⏱️ Minutes", options=[0,1,15,30,45], index=1)
            st.markdown("📅 **Study frequency / duration**")
            duration_unit = st.selectbox("Duration unit", options=["Weeks","Months"], index=0)
            duration_amount = st.number_input("Duration amount", min_value=1, value=4, step=1)
            repeat_per_week = st.slider("Sessions per week 🐝", min_value=1, max_value=14, value=3)
        goals = st.text_area("🎯 Notes / learning goals (optional)", placeholder="E.g. pass exams, build projects, daily revision, etc.")
        submitted = st.form_submit_button("💾 Save profile")
        if submitted:
            if not name:
                st.error("Please enter your name")
            elif hours == 0 and minutes == 0:
                st.error("Please set a session length greater than 0")
            else:
                st.session_state.profile = {
                    'name': name, 'preferred_time': preferred_time, 'subject': subject,
                    'hours': hours, 'minutes': minutes, 'duration_amount': duration_amount,
                    'duration_unit': duration_unit, 'sessions_per_week': repeat_per_week, 'goals': goals
                }
                # Do NOT start timer here. Timer will start when the user clicks "Open & Start" for a resource.
                total_seconds = int((hours * 3600) + (minutes * 60))
                if total_seconds <= 0:
                    total_seconds = 30*60
                st.session_state.time_left = total_seconds
                st.session_state.timer_running = False
                st.success("Profile saved. Go to Paths → generate a path and click a resource 'Open & Start' to begin a session.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<strong>💡 Quick tips</strong>", unsafe_allow_html=True)
    st.write("• Short, consistent sessions work best (25–60 mins).")
    st.write("• Keep goals specific and measurable.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Timer tab ----------
with tab_timer:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    profile = st.session_state.profile
    if profile:
        st.markdown(f"<h1>Welcome back, {profile['name']}! 🐿️</h1>", unsafe_allow_html=True)
        st.markdown(f"<div class='muted'>Today's session — {profile['subject']} ☕</div>", unsafe_allow_html=True)
    else:
        st.markdown("<h1>Timer</h1>", unsafe_allow_html=True)
        st.info("Create a profile in Onboarding, then generate a path and click a resource to start the timer.")

    if profile:
        if st.session_state.timer_running and st.session_state.start_time:
            elapsed = (datetime.now() - st.session_state.start_time).total_seconds()
            current_time_left = max(0, int(st.session_state.time_left - int(elapsed)))
            if current_time_left == 0 and not st.session_state.show_celebration:
                st.session_state.show_celebration = True
                st.session_state.timer_running = False
                st.balloons()
        else:
            current_time_left = st.session_state.paused_time if st.session_state.paused_time > 0 else st.session_state.time_left

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if not st.session_state.show_celebration:
            st.markdown(f"<div class='timer-display'>⏰ {format_time(current_time_left)}</div>", unsafe_allow_html=True)
            status = "🟢 In Progress" if st.session_state.timer_running else "⏸️ Paused"
            st.markdown(f"<p style='text-align:center; color:#a0522d'>{status}</p>", unsafe_allow_html=True)
            total_seconds = int((profile['hours'] * 3600) + (profile['minutes'] * 60))
            total_seconds = total_seconds if total_seconds>0 else 30*60
            progress = (total_seconds - current_time_left) / total_seconds if total_seconds>0 else 0.0
            st.progress(progress)
            c1,c2,c3 = st.columns([1,2,1])
            with c2:
                sub1, sub2 = st.columns(2)
                with sub1:
                    if st.button("⏸️ Pause" if st.session_state.timer_running else "▶️ Resume", use_container_width=True):
                        pause_timer()
                        safe_rerun()
                with sub2:
                    if st.button("🔄 Reset", use_container_width=True):
                        reset_timer()
                        safe_rerun()
        else:
            # Celebration + feedback form
            st.markdown(f"""
                <div class='celebration'>
                    <div style='font-size:3.5rem;'>🏆</div>
                    <h1 style='color:#8b4513;'>🎉 You Did It! 🎉</h1>
                    <p>You've completed your session!</p>
                </div>
            """, unsafe_allow_html=True)

            # Show which topic/resource was active
            ct = st.session_state.current_topic or "the module"
            cr = st.session_state.current_resource or "the resource"
            st.markdown(f"### How was **{ct}** for you?")

            # Emoji feedback (one-click)
            emoji_options = [
                ("very_happy", "😃", "Loved it"),
                ("happy", "🙂", "Good"),
                ("neutral", "😐", "Okay"),
                ("frustrated", "😣", "Struggled"),
                ("very_sad", "😭", "Too hard")
            ]
            cols = st.columns(len(emoji_options))
            selected_emoji_key = None
            for idx, (key, emoji, label) in enumerate(emoji_options):
                with cols[idx]:
                    # create unique key for button
                    btn_key = f"emoji_{key}_{int(datetime.utcnow().timestamp())}"
                    if st.button(emoji + " " + label, key=btn_key):
                        selected_emoji_key = key
                        st.session_state._selected_emoji = key  # temp store
                        # we don't safe_rerun yet, keep on same screen so user can answer difficulty question

            # Difficulty question
            st.markdown("### Was the module too hard for you?")
            diff_choice = st.radio("", options=["No — it was okay", "Yes — it was too hard"], index=0)

            if st.button("Submit feedback"):
                chosen = st.session_state.pop("_selected_emoji", None)
                emoji_label = next((e for k,e,_ in emoji_options if k==chosen), None)
                feedback = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "topic": st.session_state.current_topic,
                    "resource": st.session_state.current_resource,
                    "emoji_key": chosen,
                    "emoji": emoji_label,
                    "too_hard": True if diff_choice.startswith("Yes") else False
                }
                st.session_state.feedbacks.append(feedback)
                st.success("Thanks — feedback recorded!")
                # clear current topic/resource (consider module done)
                st.session_state.current_topic = None
                st.session_state.current_resource = None
                st.session_state.show_celebration = False
                safe_rerun()

            # Option to skip feedback
            if st.button("Skip feedback"):
                st.session_state.current_topic = None
                st.session_state.current_resource = None
                st.session_state.show_celebration = False
                safe_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Paths tab ----------
with tab_paths:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🛠️ Personalized Path Generator")
    st.markdown("<div class='small-muted'>Topics are loaded internally (no upload). Use the resource buttons to open a resource and start the timer. Feedback is requested when a session finishes.</div>", unsafe_allow_html=True)

    # Data-cleaning summary for integrated dataset
    cleaned, issues = data_cleaning_report(_default_df)
    st.session_state.cleaned_df = cleaned
    st.session_state.data_issues = issues
    if issues:
        st.warning(f"Data issues detected: {len(issues)} (see details)")
        with st.expander("View data-cleaning summary"):
            for it in issues:
                st.write("-", it)
            csv_bytes = cleaned.to_csv(index=False).encode('utf-8')
            st.download_button("Download cleaned CSV", data=csv_bytes, file_name="cleaned_topics.csv", mime="text/csv")
    else:
        st.success("Integrated dataset passed basic checks.")

    left_col, right_col = st.columns([2,1])
    with left_col:
        profile = st.session_state.profile
        if profile:
            interests_input = st.text_input("Interests (comma separated tags)", value=profile.get('subject',''))
        else:
            interests_input = st.text_input("Interests (comma separated tags)", value="")
        interests = [s.strip() for s in interests_input.split(',') if s.strip()]
        skill_level = st.slider("Skill level (1 beginner - 5 advanced)", min_value=1, max_value=5, value=3)
        if profile:
            target_weeks = profile['duration_amount'] if profile['duration_unit']=="Weeks" else profile['duration_amount']*4
            hours_per_week = max(1.0, profile['hours'] + profile['minutes']/60.0) * profile['sessions_per_week']
        else:
            target_weeks = st.number_input("Target weeks", value=4, min_value=1)
            hours_per_week = st.number_input("Hours per week available", value=6.0, min_value=1.0)
    with right_col:
        resource_types = st.multiselect("Resource types to show/filter", options=["video","article","interactive","other"], default=["video","article","interactive"])
        require_resources = st.checkbox("Require topics to have chosen resource types", value=False)
        max_seed = st.number_input("Top-matching topics to seed (breadth)", min_value=1, max_value=20, value=8)

        if st.button("Generate learning path"):
            topics = load_topics_from_df(st.session_state.cleaned_df)
            if not interests and profile:
                interests = [t.strip() for t in str(profile.get('subject','')).split(',') if t.strip()]
            res = generate_path(topics, interests, skill_level=int(skill_level), hours_per_week=float(hours_per_week),
                                max_seed=int(max_seed), target_weeks=int(target_weeks), resource_types=[r.lower() for r in resource_types],
                                require_resources=bool(require_resources))
            st.session_state.last_result = res
            st.success("Path generated — scroll down for results")

    # Display results (no internal IDs)
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🛤️ Generated Path")
        st.markdown(f"**Generated at:** {res['meta']['generated_at']}")
        if res.get("cycle_detected"):
            st.warning("Cycle detected in prerequisites — ordering fallback used.")
        ordered = res['ordered']
        weeks = res['weeks']

        st.subheader("Ordered topics")
        for i, t in enumerate(ordered, start=1):
            with st.expander(f"{i}. {t['title']} — est {t.get('est_hours','?')}h"):
                st.write("Tags:", ", ".join(t.get('tags', [])))
                prereq_list = ", ".join(t.get('prereqs', []))
                st.write("Prereqs:", prereq_list if prereq_list else "None")

                # Resources: show link and an "Open & Start" button that both opens the resource and starts the timer
                rlist = t.get('resources', [])
                if rlist:
                    for r_idx, r in enumerate(rlist):
                        title = r.get('title', '')
                        rtype = r.get('type', '')
                        url = r.get('url', None)
                        if not url:
                            url = "https://www.google.com/search?q=" + quote_plus(title)
                        colA, colB = st.columns([4,1])
                        with colA:
                            st.markdown(f"- **{rtype.title()}**: [{title}]({url})")
                        with colB:
                            # disable start button if timer is already running
                            disabled = st.session_state.timer_running
                            btn_key = f"openstart_{t['id']}_{r_idx}"
                            if st.button("Open & Start", key=btn_key, disabled=disabled):
                                # set current topic/resource and start session (this opens url and starts timer)
                                st.session_state.current_topic = t['title']
                                st.session_state.current_resource = title
                                start_session_with_url(url, topic_title=t['title'], resource_title=title)

                else:
                    st.write("_No resources available for this topic_")

        st.subheader("Weekly schedule")
        if weeks:
            cols_to_show = min(len(weeks), 6)
            cols = st.columns(cols_to_show)
            for idx, w in enumerate(weeks):
                c = cols[idx % cols_to_show]
                with c:
                    st.markdown(f"### Week {idx+1}")
                    if not w['topics']:
                        st.write("_No topics scheduled this week_")
                    for tt in w['topics']:
                        note = f" ({tt.get('note')})" if tt.get('note') else ""
                        st.write(f"- {tt['title']} — {round(tt.get('scheduled_hours',0),1)}h{note}")
        else:
            st.write("No weeks generated.")

        st.subheader("Export")
        b = json.dumps(res, indent=2)
        st.download_button("Download path as JSON", data=b, file_name="study_path.json", mime="application/json")
        total_hours = sum(t.get('est_hours',0) for t in ordered)
        st.info(f"Total topics: {len(ordered)} • Estimated total hours: {total_hours:.1f} • Weeks planned: {len(weeks)}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Show stored feedbacks and allow download (admin/summary)
    st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
    if st.session_state.feedbacks:
        with st.expander("View collected feedback"):
            st.write(pd.DataFrame(st.session_state.feedbacks))
            json_bytes = json.dumps(st.session_state.feedbacks, indent=2).encode('utf-8')
            st.download_button("Download feedback (JSON)", data=json_bytes, file_name="feedback.json", mime="application/json")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Auto-refresh while timer running ----------
if st.session_state.timer_running:
    time_module.sleep(1)
    safe_rerun()
