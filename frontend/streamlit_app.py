import os
import json
import requests
import streamlit as st

BACKEND_URL = os.getenv("DM_RAG_BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Decision Memo RAG", layout="wide")
st.title("Decision Memo RAG")

# -------------------------
# Helpers
# -------------------------
def get_json(path: str):
    r = requests.get(f"{BACKEND_URL}{path}", timeout=60)
    r.raise_for_status()
    return r.json()

def post_files(path: str, files):
    # files: list of uploaded file objects
    multipart = [("files", (f.name, f.getvalue(), f.type or "text/plain")) for f in files]
    r = requests.post(f"{BACKEND_URL}{path}", files=multipart, timeout=600)
    r.raise_for_status()
    return r.json()

def post_json(path: str, payload: dict):
    r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()

def short_query_guard(s: str, min_chars: int = 6) -> bool:
    return len((s or "").strip()) >= min_chars

def render_citations(citations: list):
    if not citations:
        st.caption("No citations returned.")
        return

    st.markdown("### Citations")
    for i, c in enumerate(citations, start=1):
        title = c.get("title") or c.get("path") or "source"
        heading = c.get("heading")
        dist = c.get("distance")

        line = f"**[{i}] {title}**"
        if heading:
            line += f" — _{heading}_"
        if dist is not None:
            line += f"  \nDistance: `{dist:.4f}`" if isinstance(dist, (int, float)) else f"  \nDistance: `{dist}`"

        st.markdown(line)

        excerpt = c.get("excerpt")
        if excerpt:
            with st.expander(f"Show snippet [{i}]", expanded=False):
                st.write(excerpt)

def render_hits(hits: list):
    if not hits:
        st.caption("No hits.")
        return
    with st.expander("Hits (debug)", expanded=False):
        for i, h in enumerate(hits, start=1):
            path = h.get("title") or h.get("path") or "hit"
            dist = h.get("distance")
            st.markdown(f"**#{i}** {path}" + (f" — `{dist:.4f}`" if isinstance(dist, (int, float)) else ""))
            content = h.get("content") or ""
            if content:
                st.code(content[:600] + ("…" if len(content) > 600 else ""))


# -------------------------
# Top status bar
# -------------------------
with st.expander("Backend Connection", expanded=False):
    st.write(f"Backend URL: `{BACKEND_URL}`")
    st.caption("Set DM_RAG_BACKEND_URL to point elsewhere if needed.")

tab_ingest, tab_analyze = st.tabs(["Upload / Ingest", "Analyze (RAG / GEN)"])

# =========================
# Tab 1 — Upload / Ingest
# =========================
with tab_ingest:
    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.subheader("Upload documents")
        files = st.file_uploader(
            "Upload .md or .txt files",
            type=["md", "txt"],
            accept_multiple_files=True
        )

        ingest_btn = st.button("Submit to /ingest", type="primary", disabled=(not files))

        if ingest_btn:
            with st.status("Uploading + ingesting…", expanded=True) as status:
                try:
                    res = post_files("/ingest", files)
                    status.update(label="Ingest complete", state="complete", expanded=True)

                    st.success("Ingest completed.")
                    st.json(res)

                    # Friendly summary
                    st.markdown("### Summary")
                    st.write({
                        "files_uploaded": res.get("files_received"),
                        "files_ingested": res.get("files_ingested"),
                        "chunks_created": res.get("chunks_created"),
                        "embeddings_written": res.get("embeddings_written"),
                        "errors": len(res.get("errors", [])),
                    })

                    if res.get("errors"):
                        st.markdown("### Errors")
                        st.json(res["errors"])

                except Exception as e:
                    status.update(label="Ingest failed", state="error", expanded=True)
                    st.error(str(e))

    with col_right:
        st.subheader("Corpus status")
        refresh = st.button("Refresh stats")
        if refresh:
            st.rerun()

        try:
            stats = get_json("/corpus/stats")
            st.metric("Total docs", stats.get("total_docs", 0))
            st.metric("Total chunks", stats.get("total_chunks", 0))
            st.caption(f"Last ingest: {stats.get('last_ingest_time')}")
            with st.expander("Raw stats", expanded=False):
                st.json(stats)
        except Exception as e:
            st.warning("Could not load /corpus/stats yet.")
            st.caption(str(e))

# =========================
# Tab 2 — Analyze (RAG / GEN)
# =========================
with tab_analyze:
    mode = st.radio("Mode", ["Ask (RAG QA)", "Memo (GEN)"], horizontal=True)

    controls = st.columns([2, 1, 1, 1])
    with controls[0]:
        k = st.number_input("k", min_value=1, max_value=20, value=6, step=1)
    with controls[1]:
        include_hits = st.toggle("include_hits", value=False)
    with controls[2]:
        show_hits = st.toggle("show hits panel", value=True)
    with controls[3]:
        min_chars = st.number_input("short-query guard (chars)", min_value=0, max_value=30, value=6, step=1)

    st.divider()

    if mode == "Ask (RAG QA)":
        q = st.text_input("Question", placeholder="e.g., What are the key risks with approach X?")
        run = st.button("Run /ask", type="primary", disabled=not q)

        if run:
            if min_chars > 0 and not short_query_guard(q, min_chars=min_chars):
                st.error(f"Query too short (need ≥ {min_chars} chars).")
            else:
                payload = {"question": q, "k": int(k), "include_hits": bool(include_hits)}
                try:
                    res = post_json("/ask", payload)

                    # Expected fields per your backend behavior:
                    # answer, citations, best_distance, weak_match, gating
                    st.subheader("Answer")
                    st.markdown(res.get("answer", ""))

                    meta = st.columns(3)
                    meta[0].metric("best_distance", res.get("best_distance", None))
                    meta[1].metric("weak_match", res.get("weak_match", None))
                    meta[2].metric("gating", res.get("gating", ""))  # confident|weak|insufficient

                    render_citations(res.get("citations", []))
                    if show_hits:
                        render_hits(res.get("hits", []))

                    with st.expander("Raw response", expanded=False):
                        st.json(res)

                except Exception as e:
                    st.error(str(e))

    else:
        topic = st.text_input("Memo topic", placeholder="e.g., Should we adopt approach X for Y?")
        run = st.button("Run /memo", type="primary", disabled=not topic)

        if run:
            if min_chars > 0 and not short_query_guard(topic, min_chars=min_chars):
                st.error(f"Topic too short (need ≥ {min_chars} chars).")
            else:
                payload = {"topic": topic, "k": int(k), "include_hits": bool(include_hits)}
                try:
                    res = post_json("/memo", payload)

                    st.subheader("Memo")
                    # Prefer structured sections if backend returns them
                    sections = res.get("sections", {}) or {}

                    order = [
                        ("tldr", "TL;DR"),
                        ("options_tradeoffs", "Options / Tradeoffs"),
                        ("risks_mitigations", "Risks / Mitigations"),
                        ("open_questions", "Open Questions"),
                        ("what_would_change_my_mind", "What would change my mind"),
                    ]

                    if isinstance(sections, dict):
                        for key, title in order:
                            txt = sections.get(key)
                            if txt:
                                st.markdown(f"### {title}")
                                st.markdown(txt)   # <-- use markdown, not st.write
                    else:
                        # fallback if backend returns memo as a string
                        st.markdown(res.get("memo", ""))

                    meta = st.columns(3)
                    meta[0].metric("best_distance", res.get("best_distance", None))
                    meta[1].metric("weak_match", res.get("weak_match", None))
                    meta[2].metric("gating", res.get("gating", ""))
                    render_citations(res.get("citations", []))
                    if show_hits:
                        render_hits(res.get("hits", []))

                    with st.expander("Raw response", expanded=False):
                        st.json(res)

                except Exception as e:
                    st.error(str(e))
