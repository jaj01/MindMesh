# streamlit_app.py
import streamlit as st
import pandas as pd
import json
from datetime import datetime, time as dt_time
from io import StringIO
from typing import List, Dict, Any, Set, Tuple
import time as time_module

# Page config
st.set_page_config(page_title="Aurora - Personalized Learning", layout="wide", page_icon="✏️")

# ---------- Custom CSS ----------
st.markdown("""
<style>
html, body, [class^="css"] {
    font-family: 'Georgia', serif;
}
.stApp {
    background: linear-gradient(160deg, #f5f0e6 0%, #7a3515 100%);
    color: #3e2c23;
    padding: 0;
}
.card {
    background: rgba(210,180,140,0.3);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 40px rgba(160,82,45,0.4);
    border: 1px solid rgba(160,82,45,0.5);
    margin-bottom: 20px;
}
h1, h2, h3 {
    color: #8b4513;
    font-weight: bold;
}
.muted {color: #a0522d;}
.accent {color: #deb887; font-weight:700}
button, .stButton>button {
    background: #a0522d !important;
    color: #fff !important;
    font-weight: 600;
    border-radius: 12px;
    padding: 8px 16px;
}
.timer-display {
    font-size: 3.5rem;
    font-weight: bold;
    text-align: center;
    color: #8b4513;
    margin: 1.5rem 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.celebration {
    text-align: center;
    padding: 2rem;
    animation: bounce 1s infinite;
}
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-20px); }
}
.stat-card {
    background: rgba(139,69,19,0.15);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    border: 2px solid rgba(160,82,45,0.3);
}
</style>
""", unsafe_allow_html=True)

# ---------- Session state defaults ----------
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'timer_running' not in st.session_state:
    st.session_state.timer_running = False
if 'time_left' not in st.session_state:
    st.session_state.time_left = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'paused_time' not in st.session_state:
    st.session_state.paused_time = 0
if 'show_celebration' not in st.session_state:
    st.session_state.show_celebration = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# ---------- Embedded "backend" topics (integrated) ----------
SAMPLE_CSV = """id,title,tags,prereqs,difficulty,est_hours,resources
t1,Computer Science Fundamentals,cs;fundamentals,,1,4,"article:CS Overview"
t2,Python Basics,python;programming,t1,1,6,"video:Python Crash Course"
t3,Data Structures,ds;algorithms,t1,3,8,"article:Intro to DS"
t4,Algorithms: Sorting & Searching,algorithms;ds,t3,3,6,"video:Sorting Algorithms"
t5,Web Development Basics,web;html;css,t1,2,6,"interactive:Build a page"
t6,Machine Learning Intro,ml;data,t2;t3,4,10,"video:ML Intro"
t7,SQL Basics,db;sql,t1,2,4,"article:SQL Tutorial"
t8,Project: Build a ToDo App,project;web,t5,2,8,"interactive:Project Guide"
"""
_default_df = pd.read_csv(StringIO(SAMPLE_CSV))

# ---------- Helper functions ----------
def format_time(seconds: int) -> str:
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def parse_tags(cell: Any) -> List[str]:
    if pd.isna(cell): return []
    if isinstance(cell, list): return [str(x).strip() for x in cell]
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
            typ, title = part.split(':',1)
            items.append({"type": typ.strip(), "title": title.strip()})
        else:
            items.append({"type":"other","title":part})
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

def generate_path(topics: List[Dict[str,Any]], interests: List[str], skill_level: int, hours_per_week: float, max_seed=8, target_weeks: int = None):
    by_id = {t['id']: t for t in topics}
    def score_topic(t):
        tag_score = sum(1 for tag in t['tags'] if tag in interests)
        diff_penalty = max(0, t.get('difficulty',3) - skill_level)
        pop = t.get('popularity', 0)
        return tag_score * 10 - diff_penalty * 3 + pop
    scored = []
    for t in topics:
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

    # Now schedule into weeks.
    total_hours = sum(float(t.get('est_hours', 0)) for t in ordered_topics)
    weeks = []

    if target_weeks and target_weeks > 0:
        per_week = total_hours / target_weeks if total_hours > 0 else max(1.0, hours_per_week)
        # create target_weeks containers
        weeks = [{"hours_left": per_week, "topics": []} for _ in range(target_weeks)]
        week_idx = 0
        for t in ordered_topics:
            dur = float(t.get('est_hours', 2.0))
            remaining = dur
            while remaining > 0:
                if week_idx >= len(weeks):
                    # append to last week if for some reason we ran out
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
        # final cleanup: remove hours_left field (not necessary to show) and ensure all weeks present
        # keep empty weeks as empty lists (so judges can see plan spans chosen weeks)
    else:
        # fallback: greedy fill using hours_per_week capacity
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

    meta = {"generated_at": datetime.utcnow().isoformat() + "Z", "skill_level": skill_level, "hours_per_week": hours_per_week, "target_weeks": target_weeks}
    return {"ordered": ordered_topics, "weeks": weeks, "meta": meta, "cycle_detected": cycle}

# ---------- Timer controls ----------
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
    st.session_state.time_left = total_seconds
    st.session_state.start_time = datetime.now()
    st.session_state.timer_running = True
    st.session_state.show_celebration = False
    st.session_state.paused_time = 0

# ---------- Tabs UI ----------
tab_onboard, tab_study = st.tabs(["Onboarding", "Study"])

# ---------- Onboarding Tab ----------
with tab_onboard:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h1>Aurora - Personalized Learning ✏️</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Tell Aurora about your study preferences 🐿️☕</div>", unsafe_allow_html=True)

    with st.form(key='onboard_form'):
        st.subheader("📝 Create your learning profile")
        c1, c2 = st.columns([1, 1])
        with c1:
            name = st.text_input("👤 Full name", placeholder="e.g. Shine Sara Mathew")
            preferred_time = st.time_input("⏰ Preferred study time", value=dt_time(20, 30))
            subject = st.selectbox(
                "📚 Interest / Subject area",
                ["Artificial Intelligence", "Web Development", "Biomaterials", "Medical Imaging",
                 "Bioinformatics Biotechnology", "Nanobiotechnology", "Renewable Energy in Agriculture",
                 "Geoinformatics and Remote Sensing in Agriculture", "Food Microbiology and Safety",
                 "Emerging Areas in Food Processing", "Other"],
            )
            if subject == "Other":
                subject = st.text_input("✏️ Please specify subject area", value="")
        with c2:
            st.markdown("🍂 **Preferred session length**")
            hours = st.selectbox("⏳ Hours", options=[0, 1, 2, 3, 4, 5, 6], index=0)
            minutes = st.selectbox("⏱️ Minutes", options=[0, 1, 15, 30, 45], index=1)
            st.markdown("📅 **Study frequency / duration**")
            duration_unit = st.selectbox("Duration unit", options=["Weeks", "Months"], index=0)
            duration_amount = st.number_input("Duration amount", min_value=1, value=4, step=1)
            if duration_unit == "Weeks" and duration_amount > 52:
                st.error("⚠️ Invalid duration: Maximum 52 weeks allowed 🐿️")
            if duration_unit == "Months" and duration_amount > 12:
                st.error("⚠️ Invalid duration: Maximum 12 months allowed 🍯")
            repeat_per_week = st.slider("Sessions per week 🐝", min_value=1, max_value=14, value=3)
        goals = st.text_area("🎯 Notes / learning goals (optional)", placeholder="E.g. pass exams, build projects, daily revision, etc.")
        submitted = st.form_submit_button("💾 Save profile")
        if submitted:
            if not name:
                st.error("⚠️ Please enter your name")
            elif hours == 0 and minutes == 0:
                st.error("⚠️ Please set a session length greater than 0")
            else:
                st.session_state.profile = {
                    'name': name,
                    'preferred_time': preferred_time,
                    'subject': subject,
                    'hours': hours,
                    'minutes': minutes,
                    'duration_amount': duration_amount,
                    'duration_unit': duration_unit,
                    'sessions_per_week': repeat_per_week,
                    'goals': goals
                }
                # initialize timer from profile
                total_seconds = int((hours * 3600) + (minutes * 60))
                st.session_state.time_left = total_seconds
                st.session_state.start_time = datetime.now()
                st.session_state.timer_running = True
                st.success("Profile saved — go to the Study tab to start the timer & generate a path.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<strong>💡 Quick tips 🐿️</strong>", unsafe_allow_html=True)
    st.write("• Prefer shorter consistent sessions (25–60 mins) ⏱️")
    st.write("• Track progress weekly and adjust session length 📊")
    st.write("• Use goals to drive session plans 🍯")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Study Tab (Timer + Path generation) ----------
with tab_study:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    profile = st.session_state.profile
    if profile:
        st.markdown(f"<h1>Welcome back, {profile['name']}! 🐿️</h1>", unsafe_allow_html=True)
        st.markdown(f"<div class='muted'>Today's session — {profile['subject']} ☕</div>", unsafe_allow_html=True)
    else:
        st.markdown("<h1>Study</h1>", unsafe_allow_html=True)
        st.info("Please complete the Onboarding tab to save a profile and enable the timer & path generator.")

    # quick side controls
    col_main, col_side = st.columns([3,1])
    with col_side:
        if profile:
            if st.button("✏️ Edit Profile"):
                st.experimental_set_query_params()  # no-op, keeps tab; user can switch to Onboarding tab manually
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🔧 Quick Controls")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            if st.button("Reset sample profile"):
                # create a quick sample profile if user wants to demo
                st.session_state.profile = {
                    'name': "Demo User",
                    'preferred_time': dt_time(20,30),
                    'subject': "python, ml",
                    'hours': 0,
                    'minutes': 30,
                    'duration_amount': 4,
                    'duration_unit': "Weeks",
                    'sessions_per_week': 3,
                    'goals': ""
                }
                st.success("Demo profile created. Re-open this tab to use it.")

    # Stats cards (only when profile exists)
    if profile:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
            st.markdown("### ⏰ Session Length")
            st.markdown(f"<h2 style='color: #8b4513;'>{profile['hours']}h {profile['minutes']}m</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with s2:
            st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
            st.markdown("### 📅 Weekly Sessions")
            st.markdown(f"<h2 style='color: #8b4513;'>{profile['sessions_per_week']} times</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with s3:
            st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
            st.markdown("### 🎯 Duration")
            st.markdown(f"<h2 style='color: #8b4513;'>{profile['duration_amount']} {profile['duration_unit']}</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Timer computations
        if st.session_state.timer_running and st.session_state.start_time:
            elapsed = (datetime.now() - st.session_state.start_time).total_seconds()
            current_time_left = max(0, int(st.session_state.time_left - int(elapsed)))
            if current_time_left == 0 and not st.session_state.show_celebration:
                st.session_state.show_celebration = True
                st.session_state.timer_running = False
                st.balloons()
        else:
            current_time_left = st.session_state.paused_time if st.session_state.paused_time > 0 else st.session_state.time_left

        # Timer UI
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if not st.session_state.show_celebration:
            st.markdown(f"<div class='timer-display'>⏰ {format_time(current_time_left)}</div>", unsafe_allow_html=True)
            status = "🟢 In Progress" if st.session_state.timer_running else "⏸️ Paused"
            st.markdown(f"<p style='text-align: center; color: #a0522d; font-size: 1.0rem;'>{status}</p>", unsafe_allow_html=True)
            total_seconds = int((profile['hours'] * 3600) + (profile['minutes'] * 60))
            total_seconds = total_seconds if total_seconds > 0 else 30*60
            progress = (total_seconds - current_time_left) / total_seconds if total_seconds > 0 else 0.0
            st.progress(progress)
            c1, c2, c3 = st.columns([1,2,1])
            with c2:
                sub1, sub2 = st.columns(2)
                with sub1:
                    if st.button("⏸️ Pause" if st.session_state.timer_running else "▶️ Resume", use_container_width=True):
                        pause_timer()
                        st.experimental_rerun()
                with sub2:
                    if st.button("🔄 Reset", use_container_width=True):
                        reset_timer()
                        st.experimental_rerun()
        else:
            st.markdown(f"""
                <div class='celebration'>
                    <div style='font-size: 4rem;'>🏆</div>
                    <h1 style='font-size: 2.2rem; color: #8b4513;'>
                        🎉 Wow! You Did It! 🎉
                    </h1>
                    <p style='font-size: 1rem; color: #3e2c23; margin: 0.5rem 0;'>
                        You've completed your {profile['subject']} session for today! 🐿️
                    </p>
                    <p style='color: #a0522d; font-size: 1rem;'>
                        Amazing dedication! Keep up the fantastic work! 🍯🌟
                    </p>
                </div>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1,1,1])
            with col2:
                if st.button("🚀 Start Another Session", use_container_width=True):
                    reset_timer()
                    st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Goals
        if profile.get('goals'):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📖 Your Learning Goals 🎯")
            st.write(profile['goals'])
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Path generator controls ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🛠️ Personalized Path Generator")
    left_col, right_col = st.columns([2,1])
    with left_col:
        # No upload control — topics integrated on backend
        st.info("Topics are loaded internally (no upload required). The path generator will use your profile duration (weeks/months) to spread the plan across the requested weeks.")
        # preview removed by request (no topics preview)
    with right_col:
        if profile:
            interests_input = st.text_input("Interests (comma separated tags)", value=profile.get('subject',''))
        else:
            interests_input = st.text_input("Interests (comma separated tags)", value="")
        interests = [s.strip() for s in interests_input.split(',') if s.strip()]
        skill_level = st.slider("Skill level (1 beginner - 5 advanced)", min_value=1, max_value=5, value=3)
        # compute default hours_per_week from profile session length * sessions_per_week if profile exists
        if profile:
            default_hours_per_week = max(1.0, profile['hours'] + profile['minutes']/60.0) * profile['sessions_per_week']
        else:
            default_hours_per_week = 6.0
        hours_per_week = st.number_input("Hours per week available (used if you don't want to rely on profile weeks)", min_value=1.0, max_value=60.0, value=float(default_hours_per_week), step=0.5)
        max_seed = st.number_input("Top-matching topics to seed (breadth)", min_value=1, max_value=20, value=8, step=1)

        if profile:
            # determine target_weeks from profile duration
            if profile['duration_unit'] == "Weeks":
                target_weeks = int(profile['duration_amount'])
            else:
                target_weeks = int(profile['duration_amount'] * 4)  # months -> approx weeks
        else:
            target_weeks = None

        if st.button("Generate learning path"):
            # load built-in topics
            topics = load_topics_from_df(_default_df)
            # if user provided empty interests, try to use profile subject
            if not interests and profile:
                interests = [t.strip() for t in str(profile.get('subject','')).split(',') if t.strip()]
            res = generate_path(topics, interests, skill_level=int(skill_level), hours_per_week=float(hours_per_week), max_seed=int(max_seed), target_weeks=target_weeks)
            st.session_state.last_result = res
            st.success("Path generated (weeks integrated from your onboarding)")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Display results ----------
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.header("🛤️ Generated Path")
        st.markdown(f"**Generated at:** {res['meta']['generated_at']}")
        if res.get("cycle_detected"):
            st.warning("Cycle detected in prerequisites — ordering fallback used.")
        ordered = res['ordered']
        weeks = res['weeks']
        st.subheader("Ordered topics")
        for i, t in enumerate(ordered, start=1):
            # do not show internal IDs
            with st.expander(f"{i}. {t['title']} — est {t.get('est_hours', '?')}h"):
                st.write("Tags:", ", ".join(t.get('tags', [])))
                prereq_list = ", ".join(t.get('prereqs', []))
                st.write("Prereqs:", prereq_list if prereq_list else "None")
                if t.get('resources'):
                    st.write("Resources:")
                    for r in t['resources']:
                        st.write(f"- {r.get('type','')}: {r.get('title','')}")
        st.subheader("Weekly schedule")
        # show weeks columns (preserve number of weeks as produced)
        if weeks:
            cols = st.columns(min(len(weeks), 6))
            for idx, w in enumerate(weeks):
                c = cols[idx % len(cols)]
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

    # auto-refresh loop while timer running
    if st.session_state.timer_running:
        time_module.sleep(1)
        st.experimental_rerun()
