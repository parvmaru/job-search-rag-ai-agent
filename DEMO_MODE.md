# DEMO MODE Guide

## What is DEMO_MODE?

DEMO_MODE allows the Job Search RAG AI Agent to run **without Ollama** for demonstration purposes. This is perfect for:

- 🎯 **Showcasing to stakeholders** who don't have Ollama installed
- 🌐 **Deploying to Streamlit Cloud** or other cloud platforms without LLM setup
- 🧪 **Testing UI components** independently of the LLM pipeline
- 📱 **Creating shareable demo links** that work immediately

## How to Enable DEMO_MODE

### Option 1: Environment Variable (Recommended)

```bash
# Windows PowerShell
$env:DEMO_MODE="true"
streamlit run app.py

# Windows CMD
set DEMO_MODE=true
streamlit run app.py

# macOS/Linux
export DEMO_MODE=true
streamlit run app.py
```

### Option 2: Direct Code Modification

Edit `app.py` and change:

```python
DEMO_MODE = os.getenv('DEMO_MODE', 'false').lower() == 'true'
```

to:

```python
DEMO_MODE = True  # Enable demo mode
```

## What Happens in DEMO_MODE?

### ✅ What Works

- **All UI features** remain fully functional
- **Match Score Analysis** - Shows realistic demo score (72%)
- **Tech Keywords Analysis** - Displays matched and missing keywords
- **Bullet Rewrites** - Shows 5 example rewrites in XYZ format
- **Interview Questions** - Generates 10 sample questions
- **Analytics Dashboard** - All charts and visualizations work
- **ATS Optimizer** - Full functionality with demo data

### 🎭 Mock Data Provided

- **Match Score**: 72/100 (Good match)
- **Matched Keywords**: python, javascript, react, node.js, postgresql, mongodb, git, rest, api, agile
- **Missing Keywords**: kubernetes, docker, terraform, aws, microservices, graphql, redis, elasticsearch
- **Bullet Rewrites**: 5 realistic examples in XYZ format
- **Interview Questions**: 10 technical and behavioral questions
- **Suggested Projects**: 3 project ideas with technologies

## Deploying to Streamlit Cloud (Shareable Link)

### Step 1: Prepare Your Repository

1. Ensure `DEMO_MODE` is set via environment variable (not hardcoded)
2. Create `requirements.txt` (already exists)
3. Create `.streamlit/config.toml` (optional, for app configuration)

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `parvmaru/job-search-rag-ai-agent`
5. Set branch to `main`
6. Set main file to `app.py`
7. **Add environment variable**: `DEMO_MODE` = `true`
8. Click "Deploy"

### Step 3: Share Your Link

Once deployed, you'll get a shareable link like:
```
https://your-app-name.streamlit.app
```

Share this link with anyone to show the working application!

## Environment Variables for Streamlit Cloud

In Streamlit Cloud settings, add:

| Variable | Value | Description |
|----------|-------|-------------|
| `DEMO_MODE` | `true` | Enables demo mode (no Ollama required) |
| `OLLAMA_HOST` | `http://localhost:11434` | (Not needed in demo mode) |
| `MODEL` | `llama3.2` | (Not needed in demo mode) |

## Switching Between Modes

### Enable Demo Mode
```bash
export DEMO_MODE=true  # or set in Streamlit Cloud
streamlit run app.py
```

### Disable Demo Mode (Production)
```bash
export DEMO_MODE=false  # or unset the variable
streamlit run app.py
# Make sure Ollama is running: ollama serve
```

## Demo Mode Indicator

When DEMO_MODE is active, you'll see a blue info banner at the top:

> 🎭 **DEMO MODE ACTIVE** - Using mock responses. No Ollama required. Set DEMO_MODE=false to use real AI analysis.

## Limitations in Demo Mode

- ❌ **No real AI analysis** - All responses are pre-generated mock data
- ❌ **Not personalized** - Results don't change based on your actual resume/JD
- ❌ **Static responses** - Same mock data every time

## Production Mode (DEMO_MODE = False)

When DEMO_MODE is disabled:
- ✅ Uses real Ollama LLM for analysis
- ✅ Personalized results based on your resume and JD
- ✅ Dynamic AI-generated responses
- ⚠️ Requires Ollama to be running locally

## Troubleshooting

### Demo Mode Not Working?

1. Check environment variable is set correctly:
   ```bash
   echo $DEMO_MODE  # Should output "true"
   ```

2. Verify in app.py that DEMO_MODE is being read:
   ```python
   print(f"DEMO_MODE: {DEMO_MODE}")  # Add this temporarily
   ```

3. Restart Streamlit after changing environment variables

### Streamlit Cloud Deployment Issues?

1. Ensure `requirements.txt` includes all dependencies
2. Check that `DEMO_MODE=true` is set in Streamlit Cloud secrets
3. Verify `app.py` is the main file
4. Check deployment logs for errors

## Best Practices

1. **For Demos**: Always use `DEMO_MODE=true` for consistent, fast results
2. **For Development**: Use `DEMO_MODE=false` to test real AI functionality
3. **For Production**: Use `DEMO_MODE=false` with proper Ollama setup
4. **For Sharing**: Deploy to Streamlit Cloud with `DEMO_MODE=true`

---

**Need Help?** Check the main README.md or open an issue on GitHub.

