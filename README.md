# Job Search RAG AI Agent

> A privacy-first, local AI copilot that intelligently compares resumes with job descriptions using Retrieval-Augmented Generation (RAG). No cloud APIs, no data sharing—everything runs on your machine.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-green.svg)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Ollama-LLaMA%203.2-orange.svg)](https://ollama.ai/)

---

## 🎯 Project Overview

**Job Search RAG AI Agent** is an intelligent resume analysis tool that helps job seekers optimize their resumes for specific job descriptions. Unlike cloud-based solutions, this application runs entirely on your local machine, ensuring complete privacy and data security.

### What It Does

- **Compares** your resume against job descriptions using AI-powered semantic analysis
- **Identifies** technical skill gaps and missing keywords
- **Suggests** resume improvements with AI-generated bullet rewrites
- **Generates** tailored interview questions based on job requirements
- **Optimizes** resumes for Applicant Tracking Systems (ATS)
- **Provides** visual analytics and match scoring

### Why It's Different

✅ **100% Local & Private** - No data leaves your computer  
✅ **No API Costs** - Uses free, open-source Ollama LLM  
✅ **Tech Keyword Focused** - Prioritizes technical skills over generic keywords  
✅ **RAG-Powered** - Retrieves relevant context for accurate analysis  
✅ **Production-Ready** - Clean code, error handling, caching, and modern UI

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                          │
│  (Match Score | Keywords | Bullet Rewrites | Analytics)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Job Agent (RAG Pipeline)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Retrieve   │→ │  Extract     │→ │  Generate   │      │
│  │  (FAISS)    │  │  (LLM)       │  │  (LLM)      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌───────────────┐            ┌───────────────┐
│  FAISS Index  │            │  Ollama LLM   │
│  (Vector DB) │            │  (Local)      │
└───────────────┘            └───────────────┘
        │                             │
        ▼                             ▼
┌───────────────┐            ┌───────────────┐
│  PDFs (JD +   │            │  LLaMA 3.2     │
│  Resume)      │            │  Model         │
└───────────────┘            └───────────────┘
```

### How RAG Works in This Project

**Retrieval-Augmented Generation (RAG)** combines the best of both worlds:

1. **Retrieval Phase**: 
   - PDFs (resumes & job descriptions) are split into chunks
   - Each chunk is converted to a vector embedding using Sentence Transformers
   - Vectors are stored in FAISS for fast similarity search
   - When analyzing, relevant chunks are retrieved based on semantic similarity

2. **Augmentation Phase**:
   - Retrieved chunks provide context to the LLM
   - LLM uses this context (not just its training data) to generate accurate, job-specific analysis

3. **Generation Phase**:
   - Ollama (LLaMA 3.2) generates recommendations, bullet rewrites, and interview questions
   - All responses are grounded in the actual job description and resume content

**Why RAG?** Traditional LLMs rely only on training data. RAG ensures the AI analyzes YOUR specific resume and job description, not generic advice.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Interactive web UI |
| **LLM** | Ollama (LLaMA 3.2) | Local language model for analysis |
| **Vector Store** | FAISS (CPU) | Fast similarity search for RAG |
| **Embeddings** | Sentence Transformers | Convert text to vectors |
| **PDF Processing** | PyPDF | Extract text from PDFs |
| **Analytics** | Plotly, Pandas | Visualizations and data analysis |
| **Language** | Python 3.8+ | Core implementation |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- Git (for cloning)

### Step 1: Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai/), then pull the model:

```bash
ollama pull llama3.2
```

### Step 2: Clone Repository

```bash
git clone https://github.com/parvmaru/job-search-rag-ai-agent.git
cd job-search-rag-ai-agent
```

### Step 3: Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment (Optional)

```bash
# Copy example env file
cp .env.example .env

# Edit .env if needed (defaults work for most setups)
# OLLAMA_HOST=http://localhost:11434
# MODEL=llama3.2
```

### Step 6: Start Ollama Server

Open a new terminal and run:

```bash
ollama serve
```

Keep this terminal open. Wait for "Ollama is running" message.

### Step 7: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Step 8: Build Index

1. Upload your resume and job description PDFs via the UI
2. Click "Build / Rebuild Index" in the sidebar
3. Wait for indexing to complete

### Step 9: Analyze

1. Select a Job Description and Resume from dropdowns
2. Click "Compare Resume vs Selected JD"
3. Explore results across tabs:
   - **Match Score**: Overall fit percentage
   - **Missing Keywords**: Tech keywords to add
   - **Bullet Rewrites**: AI-suggested improvements
   - **Interview Questions**: Tailored questions
   - **Analytics**: Visual skill gap analysis
   - **ATS Optimizer**: Keyword density optimization

---

## 💡 Example Use Cases

### For Job Seekers

- **"How well does my resume match this Software Engineer role?"**
  - Get a match score and specific skill gaps

- **"What technical keywords am I missing?"**
  - See prioritized list of tech keywords to add

- **"Rewrite my resume bullets to match this JD"**
  - Get AI-generated bullet points in XYZ format

- **"What interview questions should I prepare for?"**
  - Generate role-specific technical and behavioral questions

### For Recruiters

- **"Analyze candidate resumes against job requirements"**
  - Quickly identify top matches and skill gaps

- **"Optimize job descriptions for better candidate matching"**
  - Understand which keywords candidates are likely to have

---

## 🌟 Why This Project is Impressive

### Technical Excellence

- **RAG Implementation**: Properly implements Retrieval-Augmented Generation with FAISS vector search
- **Multi-Step Pipeline**: Structured analysis pipeline (retrieve → extract → compute → generate)
- **Tech Keyword Focus**: Advanced keyword extraction prioritizing technical skills over generic terms
- **Error Handling**: Robust error handling with fallbacks and user-friendly messages
- **Caching**: Implements result caching for faster repeated analyses
- **Production Code**: Clean, modular architecture with separation of concerns

### Privacy & Security

- **100% Local**: No data sent to external APIs or cloud services
- **No API Costs**: Uses free, open-source Ollama instead of paid services
- **Data Ownership**: All PDFs and analysis stay on your machine
- **Environment Variables**: Secure configuration management

### User Experience

- **Modern UI**: Dark theme, responsive design, intuitive navigation
- **Visual Analytics**: Interactive charts and graphs for skill gap analysis
- **Real-Time Feedback**: Progress indicators and status messages
- **Comprehensive Features**: Match scoring, keyword analysis, bullet rewrites, interview prep, ATS optimization

### Engineering Best Practices

- **Version Control**: Proper Git workflow with meaningful commits
- **Documentation**: Comprehensive README and code comments
- **Dependency Management**: Clean requirements.txt with pinned versions
- **Code Organization**: Modular structure (agent, analytics, llm, rag)
- **Testing Ready**: Structure supports easy unit testing

---

## 📊 Features in Detail

### 1. Match Score Analysis
- Tech keyword-based scoring (0-100)
- Weighted algorithm prioritizing must-have skills
- Visual progress bars and color-coded interpretations

### 2. Tech Keywords Analysis
- **Matched Keywords**: Green badges showing skills present in both
- **Missing Keywords**: Red badges showing skills to add
- Strict filtering to show only technical keywords (no generic words)

### 3. Bullet Rewrites
- AI-generated resume bullet points in XYZ format
- Context-aware rewrites using JD phrases
- Quantified impact suggestions

### 4. Interview Questions
- Technical questions based on job requirements
- Behavioral questions aligned with role expectations
- Role-specific scenarios

### 5. Analytics Dashboard
- Skill overlap percentages
- Donut charts for match visualization
- Bar charts for top missing keywords
- Frequency analysis

### 6. ATS Optimizer
- Keyword density analysis
- Suggested sections to add
- Top keywords ranked by importance
- Visual comparison (present vs. missing)

---

## 🔮 Future Improvements

### Planned Features

- [ ] **Multi-format Support**: Support for DOCX, TXT, and Markdown resumes
- [ ] **Batch Analysis**: Compare one resume against multiple JDs simultaneously
- [ ] **Resume Templates**: AI-generated resume templates optimized for specific roles
- [ ] **Cover Letter Generator**: Generate tailored cover letters based on JD analysis
- [ ] **Interview Prep Mode**: Simulated interviews with AI-generated follow-up questions
- [ ] **Export Reports**: PDF/HTML export of analysis reports
- [ ] **Integration**: LinkedIn profile import, job board APIs
- [ ] **Advanced Analytics**: Trend analysis across multiple applications
- [ ] **Custom Models**: Fine-tuned models for specific industries
- [ ] **Collaboration**: Share analysis reports with mentors/career coaches

### Technical Enhancements

- [ ] **GPU Support**: FAISS GPU acceleration for faster indexing
- [ ] **Model Selection**: Support for multiple Ollama models (Mistral, CodeLlama, etc.)
- [ ] **Incremental Indexing**: Update index without full rebuild
- [ ] **API Mode**: REST API for programmatic access
- [ ] **Docker Support**: Containerized deployment
- [ ] **Unit Tests**: Comprehensive test coverage
- [ ] **CI/CD**: Automated testing and deployment

---

## 📁 Project Structure

```
job-search-rag-ai-agent/
├── agent/              # RAG agent with multi-step pipeline
│   ├── __init__.py
│   └── job_agent.py    # Core analysis logic
├── analytics/          # Analysis modules
│   ├── __init__.py
│   ├── skill_gap.py    # Keyword extraction & matching
│   ├── ats_optimizer.py # ATS optimization
│   ├── detailed_analysis.py # LLM-based detailed analysis
│   └── analysis_cache.py # Result caching
├── llm/               # LLM integration
│   ├── __init__.py
│   └── ollama_client.py # Ollama API client
├── rag/               # RAG components
│   ├── __init__.py
│   ├── ingest.py       # PDF processing & indexing
│   └── retrieve.py     # FAISS retrieval
├── data/              # PDF storage (gitignored)
├── db/                # FAISS index (gitignored)
├── app.py             # Streamlit UI
├── requirements.txt   # Python dependencies
├── .env.example       # Environment template
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**parvmaru**

- GitHub: [@parvmaru](https://github.com/parvmaru)
- Project Link: [https://github.com/parvmaru/job-search-rag-ai-agent](https://github.com/parvmaru/job-search-rag-ai-agent)

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for providing local LLM infrastructure
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Streamlit](https://streamlit.io/) for the amazing UI framework

---

## ⚠️ Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
ollama list

# If not running, start it
ollama serve

# Verify model is installed
ollama pull llama3.2
```

### FAISS Import Errors

If you see "DLL load failed" on Windows:
1. Increase Windows paging file size
2. Reinstall: `pip uninstall faiss-cpu && pip install faiss-cpu --no-cache-dir`
3. Restart your computer

### Low Match Scores

- Ensure you're comparing against the correct JD
- Check that resume contains relevant technical keywords
- Review missing keywords tab for specific gaps

---

**⭐ If you find this project helpful, please consider giving it a star!**

