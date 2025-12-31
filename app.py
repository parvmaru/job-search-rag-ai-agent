"""
Streamlit UI - Job Agent Copilot
Resume Coach Interface with structured analysis
"""

import streamlit as st
from pathlib import Path
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from rag.ingest import PDFIngester
from rag.retrieve import FAISSRetriever
from llm.ollama_client import OllamaClient
from agent.job_agent import JobAgent
from analytics.skill_gap import compute_skill_gap, get_chunks_text
from analytics.detailed_analysis import analyze_jd_resume_match
from analytics.ats_optimizer import analyze_ats_match, create_keywords_chart
from typing import List
import re


# Page config with custom styling
st.set_page_config(
    page_title="Job Agent Copilot - Resume Coach",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main background - dark theme */
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .block-container {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* Cards - dark theme */
    .metric-card {
        background: #1e1e2e;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #fafafa;
    }
    
    /* Keyword badges */
    .keyword-badge {
        display: inline-block;
        background: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
    }
    
    /* Bullet items - dark theme */
    .bullet-item {
        background: #1e1e2e;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #4ecdc4;
        color: #fafafa;
    }
    
    /* Project items - dark theme */
    .project-item {
        background: #1e1e2e;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #ff6b6b;
        color: #fafafa;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Streamlit elements - dark theme */
    .stMarkdown {
        color: #fafafa;
    }
    .stText {
        color: #fafafa;
    }
    .stInfo {
        background-color: #1e1e2e;
        border-left: 4px solid #4ecdc4;
    }
    .stSuccess {
        background-color: #1e1e2e;
        border-left: 4px solid #4ecdc4;
    }
    .stWarning {
        background-color: #1e1e2e;
        border-left: 4px solid #ffa500;
    }
    .stError {
        background-color: #1e1e2e;
        border-left: 4px solid #ff6b6b;
    }
    
    /* Sidebar - dark theme */
    [data-testid="stSidebar"] {
        background-color: #1e1e2e;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #fafafa;
    }
    
    /* Tabs - dark theme */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e2e;
    }
    .stTabs [data-baseweb="tab"] {
        color: #fafafa;
    }
    
    /* Selectbox and inputs - dark theme */
    .stSelectbox label, .stTextInput label {
        color: #fafafa;
    }
    
    /* Expander - dark theme */
    .streamlit-expanderHeader {
        background-color: #1e1e2e;
        color: #fafafa;
    }
    .streamlit-expanderContent {
        background-color: #0e1117;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem;">💼 Job Agent Copilot</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">AI-Powered Resume Coach - Match your resume to job descriptions</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'selected_jd' not in st.session_state:
    st.session_state.selected_jd = None
if 'selected_resume' not in st.session_state:
    st.session_state.selected_resume = None
if 'selected_resume_name' not in st.session_state:
    st.session_state.selected_resume_name = None


def check_ollama():
    """Check if Ollama is running."""
    try:
        client = OllamaClient()
        if client.check_connection():
            return client
        else:
            st.error("⚠️ Ollama is not running or model 'llama3.2' not found.")
            st.info("Start Ollama: `ollama serve`")
            return None
    except Exception as e:
        st.error(f"⚠️ Cannot connect to Ollama: {e}")
        return None


def generate_interview_questions(agent: JobAgent, jd_name: str) -> List[str]:
    """Generate interview questions based on JD."""
    if agent is None or agent.retriever is None:
        return []
    
    retriever = agent.retriever
    jd_chunks = retriever.retrieve(
        "skills tools technologies responsibilities requirements qualifications experience needed",
        top_k=8,
        include_files=[jd_name] if jd_name else None,
        exclude_files=['resume.pdf']
    )
    
    if not jd_chunks:
        return []
    
    jd_text = "\n\n".join([chunk['text'] for chunk in jd_chunks[:5]])  # Use top 5 chunks
    jd_text = jd_text[:2000] if len(jd_text) > 2000 else jd_text
    
    prompt = f"""Based on this job description, generate 10 relevant interview questions that a candidate might be asked.

=== JOB DESCRIPTION ===
{jd_text}

Generate exactly 10 interview questions in this format:

INTERVIEW QUESTIONS:
1. [Question 1]
2. [Question 2]
3. [Question 3]
...
10. [Question 10]

IMPORTANT:
- Start with "INTERVIEW QUESTIONS:"
- Number each question (1-10)
- Questions should be specific to this role
- Mix technical, behavioral, and role-specific questions
- Be concise and actionable"""
    
    try:
        response = agent.llm.generate(prompt, temperature=0.7)
    except ConnectionError as e:
        # Return empty list if Ollama is not available
        return []
    except Exception as e:
        print(f"[WARNING] Error generating interview questions: {e}")
        return []
    
    # Parse questions
    questions = []
    # Look for numbered questions
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        # Match patterns like "1. Question" or "1) Question" or "- Question"
        match = re.match(r'^\d+[.)]\s*(.+)', line)
        if match:
            questions.append(match.group(1).strip())
        elif line and line.startswith('-') and len(line) > 10:
            questions.append(line[1:].strip())
        elif line and len(line) > 15 and '?' in line and not line.startswith('INTERVIEW'):
            questions.append(line)
    
    return questions[:10]  # Return up to 10 questions


# Sidebar: File list and ingestion
with st.sidebar:
    st.markdown("### 📁 Documents")
    
    data_dir = "data"
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    # JD files: Only files starting with "jd_"
    jd_files = [pdf for pdf in pdf_files if pdf.name.startswith("jd_")]
    # Resume files: Files with "resume" in name but NOT starting with "jd_"
    resume_files = [pdf for pdf in pdf_files if ("resume" in pdf.name.lower() and not pdf.name.startswith("jd_"))]
    if not resume_files:
        resume_pdf = Path(data_dir) / "resume.pdf"
        if resume_pdf.exists():
            resume_files = [resume_pdf]
    
    if pdf_files:
        st.success(f"✅ Found {len(pdf_files)} PDF(s)")
        st.caption(f"• {len(jd_files)} Job Description(s)")
        st.caption(f"• {len(resume_files)} Resume(s)")
    else:
        st.warning(f"No PDFs found in {data_dir}/")
        st.info("Add PDF files to the /data folder")
    
    st.divider()
    
    st.markdown("### 🔧 Index Management")
    
    # Check if index exists
    index_exists = os.path.exists("db/faiss.index") and os.path.exists("db/chunks.pkl")
    
    if index_exists:
        st.success("✅ Index exists")
        try:
            with open("db/chunks.pkl", 'rb') as f:
                chunks = pickle.load(f)
                st.caption(f"📊 {len(chunks)} chunks indexed")
        except:
            pass
    else:
        st.warning("⚠️ No index found")
        st.caption("Build index to start analysis")
    
    if st.button("🔨 Build / Rebuild Index", use_container_width=True, type="primary"):
        with st.spinner("Building index from PDFs..."):
            try:
                ingester = PDFIngester()
                ingester.build_index()
                st.success("✅ Index built successfully!")
                st.session_state.retriever = None
                st.session_state.agent = None  # Reset agent
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.divider()
    
    # Indexed Documents Stats
    st.markdown("### 📊 Indexed Documents")
    if index_exists:
        try:
            with open("db/chunks.pkl", 'rb') as f:
                chunks = pickle.load(f)
                # Count by source
                sources = {}
                for chunk in chunks:
                    source = chunk.get('source_file', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                
                for source, count in sorted(sources.items()):
                    doc_type = "📄" if source.startswith("jd_") else "📝"
                    st.caption(f"{doc_type} {source}: {count} chunks")
        except:
            pass
    
    st.divider()
    
    # Ollama status check
    st.markdown("### 🔌 Ollama Status")
    if st.session_state.llm_client:
        ollama_connected = st.session_state.llm_client.check_connection()
        if ollama_connected:
            st.success("✅ Ollama is running")
            st.caption(f"Model: llama3.2")
        else:
            st.error("❌ Ollama is not running")
            st.caption("Start with: `ollama serve`")
    else:
        st.warning("⚠️ Ollama not initialized")
    
    st.divider()
    st.caption("💡 Select JD and Resume in main screen, then click 'Compare' to analyze")


# Main content
# ============================================================================
# DEMO MODE: Skip index requirement - allow app to run without index
# ============================================================================
if not index_exists and not DEMO_MODE:
    st.info("👈 Build an index first using the sidebar button")
    st.stop()
elif DEMO_MODE and not index_exists:
    # In demo mode, show info but don't stop
    st.info("🎭 **DEMO MODE**: Index not required. Demo data will be shown automatically.")

# Initialize components
# ============================================================================
# DEMO MODE: Skip retriever initialization if index doesn't exist
# ============================================================================
if st.session_state.retriever is None:
    if DEMO_MODE and not index_exists:
        # In demo mode, we don't need a real retriever - set to None
        st.session_state.retriever = None
    else:
        try:
            st.session_state.retriever = FAISSRetriever()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

# ============================================================================
# DEMO MODE: Skip Ollama initialization if in demo mode
# ============================================================================
if DEMO_MODE:
    # In demo mode, we don't need Ollama - create dummy client for compatibility
    st.session_state.llm_client = None
    st.session_state.agent = None
    ollama_status = True  # Allow comparison in demo mode
    # Show demo mode indicator
    st.info("🎭 **DEMO MODE ACTIVE** - Using mock responses. No Ollama required. Set DEMO_MODE=false to use real AI analysis.")
else:
    # Production mode: Initialize Ollama as usual
    if st.session_state.llm_client is None:
        st.session_state.llm_client = check_ollama()
        if st.session_state.llm_client is None:
            st.error("⚠️ **Ollama is not running**")
            st.markdown("""
            **To start Ollama:**
            1. Open a new PowerShell/terminal window
            2. Run: `ollama serve`
            3. Wait for "Ollama is running" message
            4. Click "Rerun" button above or refresh this page
            """)
            st.stop()

    if st.session_state.agent is None:
        st.session_state.agent = JobAgent(st.session_state.retriever, st.session_state.llm_client)

    # Check Ollama connection before allowing comparison
    ollama_status = st.session_state.llm_client.check_connection() if st.session_state.llm_client else False
    if not ollama_status:
        st.warning("⚠️ **Ollama connection lost** - Please ensure Ollama is running (`ollama serve`) before comparing.")

# File Upload Section
st.markdown("### 📤 Upload Documents")
upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    st.markdown("#### Upload Job Description")
    uploaded_jd = st.file_uploader(
        "Upload a new JD PDF",
        type=['pdf'],
        key="jd_uploader",
        help="Upload a job description PDF. It will be saved to /data folder."
    )
    if uploaded_jd is not None:
        if st.button("💾 Save JD", key="save_jd"):
            # Save uploaded JD
            save_path = Path("data") / f"jd_{uploaded_jd.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded_jd.getbuffer())
            st.success(f"✅ Saved as {save_path.name}")
            st.info("🔄 Click 'Build / Rebuild Index' in sidebar to index the new JD")
            st.rerun()

with upload_col2:
    st.markdown("#### Upload Resume")
    uploaded_resume = st.file_uploader(
        "Upload a new Resume PDF",
        type=['pdf'],
        key="resume_uploader",
        help="Upload your resume PDF. It will be saved as resume.pdf in /data folder."
    )
    if uploaded_resume is not None:
        if st.button("💾 Save Resume", key="save_resume"):
            # Save uploaded resume
            save_path = Path("data") / "resume.pdf"
            with open(save_path, "wb") as f:
                f.write(uploaded_resume.getbuffer())
            st.success(f"✅ Saved as resume.pdf")
            st.info("🔄 Click 'Build / Rebuild Index' in sidebar to index the new resume")
            st.rerun()

st.divider()

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📋 Select Job Description")
    st.caption("💡 Analysis will be based on the selected JD only for accurate, specific feedback")
    if jd_files:
        selected_jd = st.selectbox(
            "Choose a job description",
            options=jd_files,
            format_func=lambda x: x.name if x else "Select JD",
            key="jd_selector"
        )
        # Clear cache if JD changed
        if 'selected_jd_name' in st.session_state and st.session_state.selected_jd_name != selected_jd.name:
            # JD changed - clear analysis cache
            if 'analysis_result' in st.session_state:
                del st.session_state.analysis_result
            if 'detailed_analysis' in st.session_state:
                del st.session_state.detailed_analysis
            if 'ats_analysis' in st.session_state:
                del st.session_state.ats_analysis
            # Clear file-based cache
            from analytics.analysis_cache import clear_specific_cache
            if st.session_state.get('selected_resume_name'):
                clear_specific_cache(selected_jd.name, st.session_state.get('selected_resume_name', 'resume.pdf'))
        
        st.session_state.selected_jd = selected_jd
        st.session_state.selected_jd_name = selected_jd.name if selected_jd else None
    else:
        st.error("No job description PDFs found. Upload a JD above or add JD files to /data folder.")
        st.stop()

with col2:
    st.markdown("### 📄 Select Resume")
    st.caption("💡 Choose which resume to analyze against the selected JD")
    # ============================================================================
    # DEMO MODE: Create demo resume file if none exist
    # ============================================================================
    if DEMO_MODE and not resume_files:
        # Create a demo resume file entry for demo mode
        from pathlib import Path
        demo_resume = Path("data/resume_demo.pdf")
        if demo_resume.exists() or True:  # Always show demo option
            resume_files = [demo_resume]
            st.info("🎭 **DEMO MODE**: Using demo resume")
    
    if resume_files:
        # Initialize selected_resume in session state
        if 'selected_resume' not in st.session_state:
            st.session_state.selected_resume = resume_files[0] if resume_files else None
        
        selected_resume = st.selectbox(
            "Choose a resume",
            options=resume_files,
            format_func=lambda x: x.name if x else "Select Resume",
            index=0 if st.session_state.selected_resume in resume_files else 0,
            key="resume_selector"
        )
        # Clear cache if resume changed
        if 'selected_resume_name' in st.session_state and st.session_state.selected_resume_name != selected_resume.name:
            # Resume changed - clear analysis cache
            if 'analysis_result' in st.session_state:
                del st.session_state.analysis_result
            if 'detailed_analysis' in st.session_state:
                del st.session_state.detailed_analysis
            if 'ats_analysis' in st.session_state:
                del st.session_state.ats_analysis
            # Clear file-based cache
            from analytics.analysis_cache import clear_specific_cache
            if st.session_state.get('selected_jd_name'):
                clear_specific_cache(st.session_state.get('selected_jd_name'), selected_resume.name)
        
        st.session_state.selected_resume = selected_resume
        st.session_state.selected_resume_name = selected_resume.name if selected_resume else None
        st.success(f"✅ {selected_resume.name} selected")
    else:
        st.warning("No resume found. Upload a resume above.")
        st.session_state.selected_resume = None
        st.session_state.selected_resume_name = None

st.divider()

# ============================================================================
# DEMO MODE: Auto-load demo results when page loads (no button click needed)
# ============================================================================
if DEMO_MODE:
    # Ensure we have demo selections (even if no files exist)
    if not st.session_state.get('selected_jd_name'):
        st.session_state.selected_jd_name = "jd_demo.pdf"
    if not st.session_state.get('selected_resume_name'):
        st.session_state.selected_resume_name = "resume_demo.pdf"
    
    # Auto-load cached demo results if not already loaded
    if not st.session_state.get('analysis_result'):
        selected_jd_name = st.session_state.get('selected_jd_name', 'jd_demo.pdf')
        selected_resume_name = st.session_state.get('selected_resume_name', 'resume_demo.pdf')
        
        # Load from cached demo data (instant, no generation)
        cached_data = load_cached_demo_data()
        if cached_data:
            # Load all cached results instantly from JSON file
            st.session_state.analysis_result = get_demo_analysis_result(selected_jd_name, selected_resume_name)
            st.session_state.interview_questions = cached_data.get('interview_questions', [])
            st.session_state.last_jd = selected_jd_name
            st.session_state.demo_analytics = cached_data.get('analytics', {})
            st.session_state.demo_ats_analysis = cached_data.get('ats_analysis', {})
            
            st.success("✅ **Demo results loaded from cache!** All data ready instantly.")
        else:
            # Fallback: Generate demo data if cache doesn't exist
            demo_result = get_demo_analysis_result(selected_jd_name, selected_resume_name)
            st.session_state.analysis_result = demo_result
            st.session_state.interview_questions = generate_interview_questions(None, selected_jd_name)
            st.session_state.last_jd = selected_jd_name
            st.warning("⚠️ Using generated demo data. Create demo_data.json for faster loading.")

# Compare button - require Ollama, JD, and Resume selection
# In DEMO_MODE, button is optional since results auto-load
compare_disabled = (not DEMO_MODE and not ollama_status) or not st.session_state.selected_jd or not st.session_state.selected_resume

if st.button(
    "🚀 Compare Resume vs Selected JD", 
    use_container_width=True, 
    type="primary",
    disabled=compare_disabled
):
    if not ollama_status:
        st.error("⚠️ Ollama is not running. Please start Ollama first.")
        st.markdown("""
        **Quick Fix:**
        1. Open PowerShell/terminal
        2. Run: `ollama serve`
        3. Wait for confirmation
        4. Click "Rerun" or refresh page
        """)
    else:
        with st.spinner(f"🔍 Analyzing resume match against {st.session_state.selected_jd_name}... This may take 1-2 minutes."):
            try:
                selected_jd_name = st.session_state.selected_jd.name if st.session_state.selected_jd else None
                selected_resume_name = st.session_state.selected_resume.name if st.session_state.selected_resume else "resume.pdf"
                prompt = f"Compare {selected_resume_name} against {selected_jd_name}"
                result = st.session_state.agent.answer_question(
                    prompt, 
                    top_k=10,  # Get chunks from selected JD
                    selected_jd_name=selected_jd_name,
                    selected_resume_name=selected_resume_name
                )
                st.session_state.analysis_result = result
                st.session_state.selected_jd_name = selected_jd_name
                st.success("✅ Analysis complete!")
                st.balloons()  # Celebration!
            except ConnectionError as e:
                st.error("🔌 **Connection Error**")
                st.error(str(e))
                st.markdown("""
                **Troubleshooting Steps:**
                1. Check if Ollama is running: Open PowerShell and run `ollama list`
                2. If not running, start it: `ollama serve`
                3. Verify model is installed: `ollama pull llama3.2`
                4. Wait 10-15 seconds after starting Ollama
                5. Refresh this page and try again
                """)
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                with st.expander("🔍 Technical Details"):
                    st.exception(e)

# Display results in tabs
if st.session_state.get('analysis_result'):
    result = st.session_state.analysis_result
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Match Score", 
        "🔍 Missing Keywords", 
        "✏️ Bullet Rewrites", 
        "❓ Interview Questions",
        "📈 Analytics",
        "🎯 ATS Optimizer"
    ])
    
    with tab1:
        st.markdown("### Match Score Analysis")
        
        # Match score with visual
        match_score = result.get('match_score', 0)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin: 0; color: #667eea; font-size: 3rem;">{match_score}%</h2>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Resume Match Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar
        st.progress(match_score / 100)
        
        # Interpretation
        if match_score >= 80:
            st.success("🎉 Excellent match! Your resume aligns well with the job requirements.")
        elif match_score >= 60:
            st.info("👍 Good match. Consider addressing missing keywords to improve your score.")
        elif match_score >= 40:
            st.warning("⚠️ Moderate match. Focus on the missing keywords and bullet rewrites below.")
        else:
            st.error("❌ Low match. Review missing keywords and suggested improvements.")
        
        # Full analysis text
        if result.get('answer'):
            with st.expander("📄 Full Analysis", expanded=False):
                st.markdown(result['answer'])
        
        # Sources
        if result.get('sources'):
            st.markdown("---")
            st.markdown("### 📚 Sources Used")
            for i, source in enumerate(result['sources'][:5], 1):
                st.caption(f"{i}. {source['source']} (page {source['page']}, similarity: {source['score']:.3f})")
    
    with tab2:
        st.markdown("### 🔍 Tech Keywords Analysis")
        st.markdown("**Keywords = Tech Keywords Only** (Programming languages, frameworks, tools, databases, cloud services, etc.)")
        
        # Show matched tech keywords first
        matched_keywords = result.get('matched_keywords', [])
        if matched_keywords:
            st.markdown("---")
            st.markdown("#### ✅ Matched Tech Keywords")
            st.markdown("These tech keywords from the job description are present in your resume:")
            for keyword in matched_keywords:
                st.markdown(f'<span class="keyword-badge" style="background-color: #4ecdc4;">{keyword}</span>', unsafe_allow_html=True)
        else:
            st.info("💡 No matched tech keywords identified yet.")
        
        # Show missing tech keywords
        st.markdown("---")
        st.markdown("#### ❌ Missing Tech Keywords")
        st.markdown("These tech keywords from the job description are missing or weak in your resume:")
        
        missing_keywords = result.get('missing_keywords', [])
        
        if missing_keywords:
            # Display as badges
            for keyword in missing_keywords:
                st.markdown(f'<span class="keyword-badge">{keyword}</span>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.info("💡 **Tip:** Add these tech keywords naturally throughout your resume, especially in skills, tools, and experience sections.")
        else:
            st.success("✅ No missing tech keywords identified! Your resume covers all the key technical requirements.")
        
        # Show in full analysis if available
        if result.get('answer') and 'MISSING' in result['answer']:
            with st.expander("📄 Detailed Tech Keywords Analysis"):
                # Extract missing keywords section
                import re
                match = re.search(r'MISSING.*?KEYWORDS:?\s*\n(.*?)(?=\n[A-Z]+:|$)', result['answer'], re.IGNORECASE | re.DOTALL)
                if match:
                    st.markdown(match.group(1))
    
    with tab3:
        st.markdown("### Bullet Rewrites (XYZ Format)")
        st.markdown("**XYZ Format:** Did **X** (what), measured by **Y** (result), by doing **Z** (how)")
        
        bullet_rewrites = result.get('bullet_rewrites', [])
        
        # Also check detailed analysis for bullet rewrites
        if not bullet_rewrites and st.session_state.get('detailed_analysis'):
            bullet_rewrites = st.session_state.detailed_analysis.get('bullet_rewrites', [])
        
        if bullet_rewrites:
            for i, bullet in enumerate(bullet_rewrites, 1):
                # Handle both string and dict formats
                if isinstance(bullet, dict):
                    bullet_text = bullet.get('rewritten', bullet.get('text', str(bullet)))
                else:
                    bullet_text = str(bullet)
                
                st.markdown(f"""
                <div class="bullet-item">
                    <strong>{i}. {bullet_text}</strong>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Try to extract from full answer
            if result.get('answer') and 'BULLET REWRITES' in result['answer']:
                import re
                match = re.search(r'BULLET REWRITES.*?:\s*\n(.*?)(?=\n[A-Z]+:|$)', result['answer'], re.IGNORECASE | re.DOTALL)
                if match:
                    bullets_text = match.group(1)
                    st.markdown(bullets_text)
                else:
                    st.info("💡 Bullet rewrites will appear here after running the analysis.")
            else:
                st.info("💡 No bullet rewrites generated yet. Run the comparison to generate bullet rewrites.")
        
        st.markdown("---")
        st.markdown("""
        **XYZ Format Example:**
        - ❌ "Worked on improving sales"
        - ✅ "Increased sales by 30%, measured by quarterly revenue reports, by implementing a new CRM system"
        """)
    
    with tab4:
        st.markdown("### Interview Questions")
        # Use selected JD from main screen
        jd_for_questions = st.session_state.selected_jd_name
        st.markdown(f"Questions tailored to **{jd_for_questions}**")
        
        if 'interview_questions' not in st.session_state or st.session_state.get('last_jd') != jd_for_questions:
            if jd_for_questions and st.session_state.agent:
                # Check Ollama connection first
                if not st.session_state.llm_client or not st.session_state.llm_client.check_connection():
                    st.warning("⚠️ Ollama is not running. Interview questions require Ollama.")
                    st.info("💡 To generate interview questions:\n1. Open PowerShell\n2. Run: `ollama serve`\n3. Wait for 'Ollama is running'\n4. Refresh this page")
                    st.session_state.interview_questions = []
                else:
                    with st.spinner(f"Generating interview questions for {jd_for_questions}..."):
                        try:
                            questions = generate_interview_questions(
                                st.session_state.agent, 
                                jd_for_questions
                            )
                            # Questions should already be a list from the updated function
                            if isinstance(questions, list) and questions:
                                st.session_state.interview_questions = questions[:10]
                            elif isinstance(questions, str):
                                # Fallback parsing if still a string
                                import re
                                question_lines = []
                                lines = questions.split('\n')
                                for line in lines:
                                    line = line.strip()
                                    # Match patterns like "1. Question" or "1) Question" or "- Question"
                                    match = re.match(r'^\d+[.)]\s*(.+)', line)
                                    if match:
                                        question_lines.append(match.group(1).strip())
                                    elif line.startswith('-') and len(line) > 10 and '?' in line:
                                        question_lines.append(line[1:].strip())
                                    elif len(line) > 15 and '?' in line and not line.startswith('INTERVIEW'):
                                        question_lines.append(line)
                                st.session_state.interview_questions = question_lines[:10] if question_lines else []
                            else:
                                st.session_state.interview_questions = []
                            st.session_state.last_jd = jd_for_questions
                        except Exception as e:
                            st.error(f"Error generating questions: {e}")
                            with st.expander("Technical Details"):
                                st.exception(e)
                            st.session_state.interview_questions = []
            else:
                st.warning("⚠️ Please select a JD to generate interview questions")
                st.session_state.interview_questions = []
        
        if st.session_state.get('interview_questions'):
            for i, question in enumerate(st.session_state.interview_questions, 1):
                st.markdown(f"""
                <div style="background: #1e1e2e; padding: 1rem; border-radius: 6px; margin: 0.5rem 0; border-left: 3px solid #667eea; color: #fafafa;">
                    <strong>Q{i}:</strong> {question}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("💡 Interview questions will be generated here.")
        
        st.markdown("---")
        st.info("💡 **Tip:** Prepare STAR (Situation, Task, Action, Result) format answers for behavioral questions.")
    
    with tab5:
        st.markdown("### 📈 Skill Gap Analytics")
        st.markdown("Fast keyword-based analysis of skills overlap and gaps")
        
        # ============================================================================
        # DEMO MODE: Use cached analytics data
        # ============================================================================
        if DEMO_MODE and st.session_state.get('demo_analytics'):
            gap_analysis = st.session_state.demo_analytics
            # Display cached analytics data
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Match Score", f"{gap_analysis.get('overlap_percentage', 72)}%")
            with col2:
                st.metric("JD Keywords", gap_analysis.get('jd_keyword_count', 25))
            with col3:
                st.metric("Resume Keywords", gap_analysis.get('resume_keyword_count', 18))
            
            # Show charts using cached data
            if gap_analysis.get('missing_keywords_with_freq'):
                st.markdown("#### Top Missing Keywords (by frequency in JD)")
                df_missing = pd.DataFrame(
                    gap_analysis['missing_keywords_with_freq'],
                    columns=['Keyword', 'Frequency']
                )
                fig_bar = px.bar(
                    df_missing.head(15),
                    x='Frequency',
                    y='Keyword',
                    orientation='h',
                    color='Frequency',
                    color_continuous_scale='Reds',
                    title='Top 15 Missing Keywords'
                )
                fig_bar.update_layout(
                    height=500,
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis_title="Frequency in Job Description",
                    yaxis_title="Keywords"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Display missing keywords list
                with st.expander("📋 All Missing Keywords"):
                    for keyword, freq in gap_analysis['missing_keywords_with_freq']:
                        st.write(f"• **{keyword}** (appears {freq} time{'s' if freq > 1 else ''} in JD)")
            
            if gap_analysis.get('present_keywords'):
                st.markdown("---")
                st.markdown("#### ✅ Present Keywords (Good Matches)")
                present_text = ", ".join(gap_analysis['present_keywords'][:30])
                st.info(present_text)
                if len(gap_analysis['present_keywords']) > 30:
                    st.caption(f"... and {len(gap_analysis['present_keywords']) - 30} more")
        # Get JD and Resume chunks for analytics (production mode)
        elif st.session_state.agent and st.session_state.selected_jd:
            with st.spinner("🔍 Computing skill gap analytics..."):
                try:
                    # Get selected JD and Resume names
                    selected_jd_name = st.session_state.selected_jd.name if st.session_state.selected_jd else None
                    selected_resume_name = st.session_state.selected_resume.name if st.session_state.selected_resume else "resume.pdf"
                    
                    # Retrieve chunks for analytics (fast, no LLM) - USE SELECTED JD ONLY
                    if selected_jd_name:
                        jd_chunks = st.session_state.retriever.retrieve(
                            "skills tools technologies responsibilities requirements",
                            top_k=15,
                            include_files=[selected_jd_name]  # FIXED: Use selected JD only
                        )
                        # Filter to ensure only selected JD
                        jd_chunks = [chunk for chunk in jd_chunks if chunk.get('source_file', '') == selected_jd_name]
                    else:
                        jd_chunks = []
                    
                    resume_chunks = st.session_state.retriever.retrieve(
                        "experience skills projects achievements",
                        top_k=15,
                        include_files=[selected_resume_name]  # FIXED: Use selected resume
                    )
                    # Filter to ensure only selected resume
                    resume_chunks = [chunk for chunk in resume_chunks if chunk.get('source_file', '') == selected_resume_name]
                    
                    # Extract text
                    jd_text = get_chunks_text(jd_chunks)
                    resume_text = get_chunks_text(resume_chunks)
                    
                    # Compute skill gap
                    gap_analysis = compute_skill_gap(jd_text, resume_text)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overlap", f"{gap_analysis['overlap_percentage']}%")
                    with col2:
                        st.metric("Missing", f"{gap_analysis['missing_percentage']}%")
                    with col3:
                        st.metric("JD Keywords", gap_analysis['jd_keyword_count'])
                    
                    # Donut chart: Overlap vs Missing
                    st.markdown("#### Overlap vs Missing Skills")
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=['Overlap', 'Missing'],
                        values=[gap_analysis['overlap_percentage'], gap_analysis['missing_percentage']],
                        hole=0.5,
                        marker_colors=['#4ecdc4', '#ff6b6b'],
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    fig_donut.update_layout(
                        height=400,
                        showlegend=True,
                        font=dict(size=14)
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                    
                    # Bar chart: Top Missing Keywords
                    if gap_analysis['missing_keywords_with_freq']:
                        st.markdown("#### Top Missing Keywords (by frequency in JD)")
                        df_missing = pd.DataFrame(
                            gap_analysis['missing_keywords_with_freq'],
                            columns=['Keyword', 'Frequency']
                        )
                        
                        fig_bar = px.bar(
                            df_missing.head(15),
                            x='Frequency',
                            y='Keyword',
                            orientation='h',
                            color='Frequency',
                            color_continuous_scale='Reds',
                            title='Top 15 Missing Keywords'
                        )
                        fig_bar.update_layout(
                            height=500,
                            yaxis={'categoryorder': 'total ascending'},
                            xaxis_title="Frequency in Job Description",
                            yaxis_title="Keywords"
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Display missing keywords list
                        with st.expander("📋 All Missing Keywords"):
                            for keyword, freq in gap_analysis['missing_keywords_with_freq']:
                                st.write(f"• **{keyword}** (appears {freq} time{'s' if freq > 1 else ''} in JD)")
                    else:
                        st.success("✅ No missing keywords found!")
                    
                    # Present keywords
                    if gap_analysis['present_keywords']:
                        st.markdown("---")
                        st.markdown("#### ✅ Present Keywords (Good Matches)")
                        present_text = ", ".join(gap_analysis['present_keywords'][:30])
                        st.info(present_text)
                        if len(gap_analysis['present_keywords']) > 30:
                            st.caption(f"... and {len(gap_analysis['present_keywords']) - 30} more")
                    
                except Exception as e:
                    st.error(f"Error computing analytics: {e}")
                    with st.expander("Technical Details"):
                        st.exception(e)
        else:
            st.info("👈 Run a comparison first to see analytics")
    
    with tab6:
        st.markdown("### 🎯 ATS Optimizer")
        st.markdown("Optimize your resume for Applicant Tracking Systems (ATS)")
        
        # Get selected JD and Resume from main screen
        analysis_jd = st.session_state.selected_jd.name if st.session_state.selected_jd else None
        analysis_resume = st.session_state.selected_resume.name if st.session_state.selected_resume else "resume.pdf"
        
        if not analysis_jd:
            st.warning("⚠️ Please select a JD from the sidebar first.")
            st.info("👈 Use the 'Analysis Tab Selectors' in the sidebar to choose your documents.")
        else:
            st.info(f"📄 **Analyzing:** JD = `{analysis_jd}` | Resume = `{analysis_resume}`")
            
            # ============================================================================
            # DEMO MODE: Use cached ATS analysis data
            # ============================================================================
            if DEMO_MODE and st.session_state.get('demo_ats_analysis'):
                ats_analysis = st.session_state.demo_ats_analysis
                
                # Display keyword density
                st.markdown("---")
                st.markdown("#### 📊 Keyword Density")
                st.metric("Overall Keyword Density", f"{ats_analysis.get('keyword_density', 3.2):.1f}%")
                st.caption("Higher density = better ATS match (aim for 2-5%)")
                
                # Display keywords to add
                keywords_to_add = ats_analysis.get('keywords_to_add', [])
                if keywords_to_add:
                    st.markdown("---")
                    st.markdown(f"#### 🔑 Top Keywords to Add (Ranked by Priority)")
                    st.markdown(f"**Top {min(20, len(keywords_to_add))} Missing Keywords:**")
                    
                    for i, kw_data in enumerate(keywords_to_add[:20], 1):
                        col1, col2, col3 = st.columns([3, 1, 2])
                        with col1:
                            st.markdown(f"**{i}. {kw_data['keyword']}**")
                        with col2:
                            priority = kw_data.get('priority', 'medium')
                            priority_color = '🔴' if priority == 'high' else '🟡' if priority == 'medium' else '🟢'
                            st.markdown(f"{priority_color} {priority.upper()}")
                        with col3:
                            st.caption(f"Freq: {kw_data.get('frequency', 0)} | Section: {kw_data.get('suggested_section', 'Skills')}")
                    
                    if len(keywords_to_add) > 20:
                        with st.expander(f"📋 View All {len(keywords_to_add)} Missing Keywords"):
                            for i, kw_data in enumerate(keywords_to_add[20:], 21):
                                st.markdown(f"{i}. **{kw_data['keyword']}** (Freq: {kw_data['frequency']}, Section: {kw_data['suggested_section']})")
                
                # Display suggested sections
                suggested_sections = ats_analysis.get('suggested_sections', [])
                if suggested_sections:
                    st.markdown("---")
                    st.markdown("#### 💡 Suggested Resume Sections")
                    for i, section in enumerate(suggested_sections, 1):
                        st.info(f"{i}. {section}")
            # Check cache for ATS analysis (production mode)
            elif 'ats_analysis' not in st.session_state or st.session_state.get('ats_cache_key') != f"ats_{analysis_jd}_{analysis_resume}":
                st.session_state.ats_cache_key = ats_cache_key
                st.session_state.ats_analysis = None
            
            # Run ATS analysis
            if st.session_state.ats_analysis is None:
                if not st.session_state.retriever:
                    st.error("⚠️ Retriever not initialized. Please rebuild index.")
                else:
                    with st.spinner("🔍 Analyzing ATS optimization... This may take a moment."):
                        try:
                            # Retrieve JD and Resume chunks
                            jd_chunks = st.session_state.retriever.retrieve(
                                "skills tools technologies responsibilities requirements qualifications",
                                top_k=15,
                                include_files=[analysis_jd] if analysis_jd else None,
                                exclude_files=[analysis_resume] if analysis_jd else [analysis_resume]
                            )
                            
                            resume_chunks = st.session_state.retriever.retrieve(
                                "experience skills projects achievements responsibilities education",
                                top_k=15,
                                include_files=[analysis_resume]
                            )
                            
                            # Filter to ensure correct files
                            if analysis_jd:
                                jd_chunks = [chunk for chunk in jd_chunks if chunk.get('source_file', '') == analysis_jd]
                            if analysis_resume:
                                resume_chunks = [chunk for chunk in resume_chunks if chunk.get('source_file', '') == analysis_resume]
                            
                            # Extract text
                            jd_text = "\n\n".join([chunk['text'] for chunk in jd_chunks])
                            resume_text = "\n\n".join([chunk['text'] for chunk in resume_chunks])
                            
                            if jd_text and resume_text:
                                # Run ATS analysis
                                ats_result = analyze_ats_match(jd_text, resume_text)
                                st.session_state.ats_analysis = ats_result
                                st.success("✅ ATS analysis complete!")
                            else:
                                st.error("❌ Could not extract text from selected files.")
                                st.session_state.ats_analysis = {
                                    'keywords_to_add': [],
                                    'suggested_sections': {},
                                    'keyword_density': {},
                                    'top_keywords_data': []
                                }
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                            with st.expander("Technical Details"):
                                st.exception(e)
                            st.session_state.ats_analysis = {
                                'keywords_to_add': [],
                                'suggested_sections': {},
                                'keyword_density': {},
                                'top_keywords_data': []
                            }
            
            # Display ATS results
            if st.session_state.ats_analysis:
                ats_result = st.session_state.ats_analysis
                
                # Keyword Density Stats
                st.markdown("---")
                st.markdown("### 📊 Keyword Density Statistics")
                density = ats_result.get('keyword_density', {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("JD Keywords", density.get('jd_total', 0))
                with col2:
                    st.metric("Resume Keywords", density.get('resume_total', 0))
                with col3:
                    st.metric("Match %", f"{density.get('match_percentage', 0):.1f}%")
                with col4:
                    st.metric("Missing", density.get('missing_count', 0))
                
                # Keywords Chart
                st.markdown("---")
                st.markdown("### 📈 Top Keywords: Present vs Missing")
                keywords_data = ats_result.get('top_keywords_data', [])
                if keywords_data:
                    chart = create_keywords_chart(keywords_data, top_n=25)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    st.caption("💡 **Legend:** 🟢 Green = Present in Resume | 🔴 Red = Missing from Resume")
                else:
                    st.info("No keyword data available.")
                
                # Keywords to Add (Ranked)
                st.markdown("---")
                st.markdown("### 🎯 Keywords to Add (Ranked by Importance)")
                keywords_to_add = ats_result.get('keywords_to_add', [])
                
                if keywords_to_add:
                    # Show top 20
                    st.markdown(f"**Top {min(20, len(keywords_to_add))} Missing Keywords:**")
                    
                    for i, kw_data in enumerate(keywords_to_add[:20], 1):
                        col1, col2, col3 = st.columns([3, 1, 2])
                        with col1:
                            st.markdown(f"**{i}. {kw_data['keyword']}**")
                        with col2:
                            st.caption(f"Freq: {kw_data['frequency']}")
                        with col3:
                            st.caption(f"→ Add to: **{kw_data['suggested_section'].title()}**")
                    
                    if len(keywords_to_add) > 20:
                        with st.expander(f"📋 View All {len(keywords_to_add)} Missing Keywords"):
                            for i, kw_data in enumerate(keywords_to_add[20:], 21):
                                st.markdown(f"{i}. **{kw_data['keyword']}** (Freq: {kw_data['frequency']}, Section: {kw_data['suggested_section']})")
                else:
                    st.success("✅ No missing keywords! Your resume covers all important keywords.")
                
                # Suggested Sections
                st.markdown("---")
                st.markdown("### 📝 Suggested Sections")
                suggested_sections = ats_result.get('suggested_sections', {})
                
                if suggested_sections:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if suggested_sections.get('skills'):
                            st.markdown("#### 💼 Skills Section")
                            skills_text = ", ".join(suggested_sections['skills'][:15])
                            st.info(skills_text)
                            if len(suggested_sections['skills']) > 15:
                                st.caption(f"... and {len(suggested_sections['skills']) - 15} more")
                        
                        if suggested_sections.get('summary'):
                            st.markdown("#### 📄 Summary Section")
                            summary_text = ", ".join(suggested_sections['summary'][:10])
                            st.info(summary_text)
                    
                    with col2:
                        if suggested_sections.get('experience'):
                            st.markdown("#### 💼 Experience Section")
                            exp_text = ", ".join(suggested_sections['experience'][:15])
                            st.info(exp_text)
                            if len(suggested_sections['experience']) > 15:
                                st.caption(f"... and {len(suggested_sections['experience']) - 15} more")
                        
                        if suggested_sections.get('projects'):
                            st.markdown("#### 🚀 Projects Section")
                            proj_text = ", ".join(suggested_sections['projects'][:10])
                            st.info(proj_text)
                            if len(suggested_sections['projects']) > 10:
                                st.caption(f"... and {len(suggested_sections['projects']) - 10} more")
                else:
                    st.info("No section suggestions available.")
                
                # Tips
                st.markdown("---")
                st.markdown("### 💡 ATS Optimization Tips")
                st.markdown("""
                1. **Natural Integration**: Add keywords naturally throughout your resume, not just in a keyword list
                2. **Context Matters**: Use keywords in context (e.g., "Developed Python applications" not just "Python")
                3. **Section Placement**: 
                   - **Skills Section**: Technical skills, tools, languages
                   - **Experience Section**: Action verbs and technologies used in past roles
                   - **Projects Section**: Technologies and frameworks used in projects
                4. **Keyword Density**: Aim for 2-3% keyword density (not too sparse, not keyword stuffing)
                5. **Variations**: Use variations of keywords (e.g., "Machine Learning" and "ML")
                """)

# Footer
st.markdown("---")
st.caption("💼 Job Agent Copilot - Powered by RAG (Retrieval-Augmented Generation) and Ollama")
