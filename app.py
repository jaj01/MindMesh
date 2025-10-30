# streamlit_app.py
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from io import StringIO
from typing import List, Dict, Any, Tuple, Set

st.set_page_config(page_title="StudyPath (Streamlit)", layout="wide")

# ---------- Utilities ----------
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
    # try JSON first
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # fallback: pipe-separated items like "video:Intro|article:Docs"
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
    for _, row in df.iterrows():
        t = {
            "id": str(row.get("id")).strip(),
            "title": str(row.get("title")),
            "tags": parse_tags(row.get("tags")),
            "prereqs": parse_prereqs(row.get("prereqs")),
            "difficulty": int(row.get("difficulty")) if not pd.isna(row.get("difficulty")) else 3,
            "est_hours": float(row.get("est_hours")) if not pd.isna(row.get("est_hours")) else 2.0,
            "resources": parse_resources(row.get("resources")) if "resources" in df.columns else []
        }
        topics.append(t)
    return topics

# Topological sort that returns order or partial order if cycles
def topo_sort_with_prereqs(topics: List[Dict[str,Any]]) -> Tuple[List[str], bool]:
    adj = {}
    for t in topics:
        adj[t['id']] = [p for p in (t.get('prereqs') or []) if p in adj or True]  # we'll filter later
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

# path generation (rule-based)
def generate_path(topics: List[Dict[str,Any]], interests: List[str], skill_level: int, hours_per_week: float, max_seed=8):
    by_id = {t['id']: t for t in topics}
    # score topics
    def score_topic(t):
        tag_score = sum(1 for tag in t['tags'] if tag in interests)
        diff_penalty = max(0, t.get('difficulty',3) - skill_level)
        pop = 0  # could be from dataset
        return tag_score * 10 - diff_penalty * 3 + pop

    scored = []
    for t in topics:
        s = score_topic(t)
        scored.append({**t, "score": s})
    scored.sort(key=lambda x: x['score'], reverse=True)

    # seed top matches, but ensure topics with any matching tag also included
    seed = []
    for t in scored:
        if len(seed) < max_seed and (t['score'] >= 0 or any(tag in interests for tag in t['tags'])):
            seed.append(t['id'])
    # include prerequisites recursively
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

    # topological order
    ordered_ids, cycle = topo_sort_with_prereqs(selected)
    if cycle:
        # fallback: simple score sort
        selected_sorted = sorted(selected, key=lambda x: -x.get('score',0))
        ordered = [t['id'] for t in selected_sorted]
    else:
        # keep only those in selected
        ordered = [oid for oid in ordered_ids if oid in by_id and oid in needed]

    ordered_topics = [by_id[i] for i in ordered]

    # schedule into weeks greedily
    weeks = []
    current_week = {"hours_left": hours_per_week, "topics": []}
    for t in ordered_topics:
        dur = float(t.get('est_hours', 2.0))
        if dur <= current_week['hours_left']:
            current_week['topics'].append({**t, "scheduled_hours": dur})
            current_week['hours_left'] -= dur
        else:
            # if current week has something, push and start new
            if current_week['topics']:
                weeks.append(current_week)
                current_week = {"hours_left": hours_per_week, "topics": []}
            # if single topic longer than a week, split across weeks
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

    meta = {"generated_at": datetime.utcnow().isoformat() + "Z", "skill_level": skill_level, "hours_per_week": hours_per_week}
    return {"ordered": ordered_topics, "weeks": weeks, "meta": meta, "cycle_detected": cycle}

# ---------- UI ----------
st.title("StudyPath — Personalized learning path (Streamlit)")

col1, col2 = st.columns([2,1])

with col1:
    st.header("Load resources (CSV)")
    uploaded = st.file_uploader("Upload CSV of topics (or leave blank to load sample)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success("CSV loaded.")
        except Exception as e:
            st.error(f"Unable to read CSV: {e}")
            st.stop()
    else:
        st.info("Using built-in sample topics. Upload your CSV to override.")
        # sample df
        sample_csv = """id,title,tags,prereqs,difficulty,est_hours,resources
t1,Computer Science Fundamentals,cs;fundamentals,,1,4,"article:CS Overview"
t2,Python Basics,python;programming,t1,1,6,"video:Python Crash Course"
t3,Data Structures,ds;algorithms,t1,3,8,"article:Intro to DS"
t4,Algorithms: Sorting & Searching,algorithms;ds,t3,3,6,"video:Sorting Algorithms"
t5,Web Development Basics,web;html;css;t1,t1,2,6,"interactive:Build a page"
t6,Machine Learning Intro,ml;data,t2; t3,4,10,"video:ML Intro"
t7,SQL Basics,db;sql;t1,2,4,"article:SQL Tutorial"
t8,Project: Build a ToDo App,project;web;t5,2,8,"interactive:Project Guide"
"""
        df = pd.read_csv(StringIO(sample_csv))

    topics = load_topics_from_df(df)

    st.subheader("Topics preview")
    # show a sample of the topic table with tags nicely rendered
    preview = pd.DataFrame([{
        "id": t["id"],
        "title": t["title"],
        "tags": ";".join(t["tags"]),
        "prereqs": ";".join(t["prereqs"]),
        "difficulty": t["difficulty"],
        "est_hours": t["est_hours"]
    } for t in topics])
    st.dataframe(preview, use_container_width=True, height=250)

with col2:
    st.header("Generate path")
    interests_input = st.text_input("Interests (comma separated tags)", value="python, ml, web")
    interests = [s.strip() for s in interests_input.split(',') if s.strip()]

    skill_level = st.slider("Skill level (1 beginner - 5 advanced)", min_value=1, max_value=5, value=3)
    hours_per_week = st.number_input("Hours per week available", min_value=1.0, max_value=60.0, value=6.0, step=0.5)
    max_seed = st.number_input("How many top-matching topics to seed (controls breadth)", min_value=1, max_value=20, value=8, step=1)

    if st.button("Generate learning path"):
        result = generate_path(topics, interests, skill_level, hours_per_week, max_seed=int(max_seed))
        st.session_state['last_result'] = result
        st.success("Path generated")

# show results if available
if 'last_result' in st.session_state:
    res = st.session_state['last_result']
    st.header("Generated Path")
    st.markdown(f"**Generated at:** {res['meta']['generated_at']}")
    if res.get("cycle_detected"):
        st.warning("Cycle detected in prerequisites — ordering fallback used.")

    ordered = res['ordered']
    weeks = res['weeks']

    st.subheader("Ordered topics")
    for i, t in enumerate(ordered, start=1):
        with st.expander(f"{i}. {t['title']} ({t['id']}) — est {t.get('est_hours', '?')}h"):
            st.write("Tags:", ", ".join(t.get('tags', [])))
            st.write("Prereqs:", ", ".join(t.get('prereqs', [])) or "None")
            if t.get('resources'):
                st.write("Resources:")
                for r in t['resources']:
                    st.write(f"- {r.get('type','')}: {r.get('title','')}")
    st.subheader("Weekly schedule")
    cols = st.columns(len(weeks) if len(weeks)<=6 else 3)
    for idx, w in enumerate(weeks):
        c = cols[idx % len(cols)]
        with c:
            st.markdown(f"### Week {idx+1}")
            for tt in w['topics']:
                note = f" — {tt.get('note')}" if tt.get('note') else ""
                st.write(f"- {tt['title']} — {tt['scheduled_hours']}h{note}")

    # export options
    st.subheader("Export / Download")
    if st.button("Download path as JSON"):
        b = json.dumps(res, indent=2)
        st.download_button("Click to download JSON", data=b, file_name="study_path.json", mime="application/json")

    # quick recap metrics
    total_hours = sum(t.get('est_hours',0) for t in ordered)
    st.info(f"Total topics: {len(ordered)} • Estimated total hours: {total_hours:.1f} • Weeks planned: {len(weeks)}")

st.markdown("---")
st.caption("Tip: upload your CSV with columns id,title,tags,prereqs,difficulty,est_hours,resources. Tags/prereqs use ';' as separator.")
