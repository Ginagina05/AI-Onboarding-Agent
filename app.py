import streamlit as st
import subprocess
import os
from pathlib import Path
from onboarding_agent import build_vectors, score_query
from openai import OpenAI
import httpx

# --- Page setup ---
st.set_page_config(page_title="AI Agent", layout="wide")

# --- Styling ---
st.markdown(
    """ 
    <style>  
        :root {
            color-scheme: light;
        }
        .header-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-container img {
            height: 50px;
        }
        .header-container h1 {
            font-size: 2rem;
            margin: 0;
            padding: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header Layout ---
col_logo, col_title = st.columns([1, 5])

with col_title:
    st.title("GINA AI Agent")

# --- Initialize OpenAI client ---
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(verify=False)
)
--- Ensure output folder ---
os.makedirs("ai_onboarding_out", exist_ok=True)

# --- Repository Input ---
repo_path = st.text_input("Enter repository path", "")

# --- Action buttons ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Index Codebase") and repo_path:
        st.info("Indexing... please wait")
        result = subprocess.run([
            "python", "onboarding_agent.py", "index",
            "--root", repo_path, "--out", "./ai_onboarding_out"
        ], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("✅ Indexing complete!")
        else:
            st.error(f"❌ Error during indexing:\n{result.stderr}")

with col2:
    if st.button("Generate Report"):
        index_file = "./ai_onboarding_out/index.jsonl"
        if os.path.exists(index_file):
            st.info("Generating report...")
            result = subprocess.run([
                "python", "onboarding_agent.py", "report",
                "--index", index_file, "--out", "./ai_onboarding_out"
            ], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("📄 Report generated successfully!")
            else:
                st.error(f"❌ Report generation failed:\n{result.stderr}")
        else:
            st.warning("⚠️ Please index the codebase first.")

with col3:
    st.info("💬 Chat with your codebase and AI assistant below.")

st.divider()

# --- Display Onboarding Report ---
report_file = "./ai_onboarding_out/onboarding_report.md"
if os.path.exists(report_file):
    st.subheader("📘 Onboarding Report")
    with open(report_file, "r", encoding="utf-8") as f:
        st.markdown(f.read())
else:
    st.info("No report available yet. Click 'Index Codebase' and 'Generate Report' first.")

st.divider()

# =============================================================
# 💬 LOCAL Q&A CHAT (repo-specific)
# =============================================================

st.subheader("💬 Local Codebase Q&A")

index_file = Path("./ai_onboarding_out/index.jsonl")

if index_file.exists():
    if 'docs' not in st.session_state:
        st.session_state.docs, st.session_state.idf = build_vectors(index_file)

    query = st.text_input("Ask a question about the codebase:")
    if query:
        hits = score_query(query, st.session_state.docs, st.session_state.idf, topk=5)
        if hits:
            st.write("### Top Matches")
            for rank, (score, dv) in enumerate(hits, 1):
                st.write(f"**[{rank}] {dv.title}** (score={score:.3f})")
                excerpt = dv.text
                if len(excerpt) > 500:
                    excerpt = excerpt[:500] + "\n... [truncated]"
                st.code(excerpt)
        else:
            st.write("No matches found. Try a different question.")
else:
    st.info("Index file not found. Please run 'Index Codebase' first to enable chat.")

st.divider()

# =============================================================
# 🌍 GENERAL AI CHAT (GPT-4.1-mini)
# =============================================================

st.subheader("🌍 General AI Chat")

# conversational memory
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.chat_message("user").write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {"role": "system", "content": "You are a helpful assistant for Software developers."},
                    *st.session_state.chat_history,
                    {"role": "user", "content": prompt}
                ],
            ) 
            reply = response.output[0].content[0].text
            st.chat_message("assistant").write(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
