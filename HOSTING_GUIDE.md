# Hosting & Deployment Guide

This guide covers multiple options for hosting your AI Document Intelligence System online so you can share a live demo link in your interview.

## 🚀 Quick Deploy Options (Ranked by Ease)

### Option 1: Streamlit Cloud (EASIEST - Recommended for Demo)

**Pros:**
- ✅ 100% Free
- ✅ Takes 5 minutes to deploy
- ✅ Beautiful interactive UI
- ✅ No credit card required
- ✅ Perfect for demos

**Steps:**

1. **Create a GitHub account** (if you don't have one)
   - Go to https://github.com and sign up

2. **Create a new repository**
   - Click "New repository"
   - Name it: `ai-document-intelligence`
   - Make it Public
   - Initialize with README

3. **Upload your files**
   - Upload all project files to the repository
   - Make sure `streamlit_app.py` is in the root directory
   - Upload `requirements.txt`
   - Upload the entire `src/` folder

4. **Sign up for Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Click "Sign up" and use your GitHub account
   - Free tier is perfect for this project

5. **Deploy your app**
   - Click "New app"
   - Select your repository: `ai-document-intelligence`
   - Main file: `streamlit_app.py`
   - Click "Deploy!"
   - Wait 2-3 minutes for it to build

6. **Get your live URL**
   - You'll get a URL like: `https://your-app.streamlit.app`
   - Share this link in your interview!

**Environment Variables (if using OpenAI):**
- In Streamlit Cloud settings, add: `OPENAI_API_KEY=your_key`
- Otherwise, it will use mock responses (which is fine for demo)

---

### Option 2: Hugging Face Spaces (EASY - Great Alternative)

**Pros:**
- ✅ Free
- ✅ ML-focused community
- ✅ Easy deployment
- ✅ Gradio or Streamlit support

**Steps:**

1. **Sign up at Hugging Face**
   - Go to https://huggingface.co and create account

2. **Create a new Space**
   - Click "Spaces" → "Create new Space"
   - Name: `ai-document-intelligence`
   - SDK: Choose "Streamlit"
   - Visibility: Public

3. **Upload files**
   - Upload `streamlit_app.py` as `app.py` (rename it)
   - Upload `requirements.txt`
   - Upload `src/` folder

4. **Wait for build**
   - Space will automatically build and deploy
   - Get URL: `https://huggingface.co/spaces/username/ai-document-intelligence`

---

### Option 3: Railway.app (MEDIUM - More Control)

**Pros:**
- ✅ $5 free credit monthly
- ✅ Runs FastAPI backend
- ✅ Custom domain support
- ✅ Database support

**Steps:**

1. **Sign up at Railway**
   - Go to https://railway.app
   - Sign up with GitHub

2. **Create new project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Connect your repository

3. **Configure deployment**
   - Railway auto-detects Python
   - Add start command: `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`
   - Set environment variables in dashboard

4. **Get public URL**
   - Railway generates URL automatically
   - Use for API or add Streamlit frontend

---

### Option 4: Render.com (MEDIUM)

**Pros:**
- ✅ Free tier available
- ✅ Auto-deploy from GitHub
- ✅ SSL certificates included

**Steps:**

1. **Sign up at Render**
   - Go to https://render.com
   - Connect GitHub account

2. **Create Web Service**
   - Click "New" → "Web Service"
   - Connect repository
   - Environment: Python 3

3. **Configure**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

4. **Deploy**
   - Click "Create Web Service"
   - Get URL: `https://your-app.onrender.com`

---

## 📋 Files Needed for Deployment

### Minimum Required Files:

```
Repository Root:
├── streamlit_app.py          # Your web interface
├── requirements.txt           # Dependencies
├── src/                       # Source code folder
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── rag_engine.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── llm_client.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   └── chunker.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
└── .streamlit/                # Streamlit config (optional)
    └── config.toml
```

### Create `.streamlit/config.toml` (Optional but Recommended):

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false
```

### Update `requirements.txt` for Cloud Deployment:

```txt
# Core AI/ML (smaller versions for cloud)
sentence-transformers==2.3.1
faiss-cpu==1.7.4
openai==1.10.0

# Data Processing (remove large ones)
numpy==1.24.3
pypdf==3.17.4

# Web Framework
streamlit==1.29.0

# Configuration
python-dotenv==1.0.0

# Logging (lighter version)
structlog==24.1.0

# Remove these for cloud to save memory:
# - langchain (heavy)
# - pandas (if not used)
# - FastAPI (if only using Streamlit)
```

---

## 🎥 Demo Preparation Tips

### Before Your Interview:

1. **Test your live link** thoroughly
   - Make sure it loads quickly
   - Test all features
   - Have sample queries ready

2. **Prepare a demo script**:
   ```
   1. Show the interface (30 sec)
   2. Load sample documents (30 sec)
   3. Run 2-3 queries showing results (1 min)
   4. Show source attribution (30 sec)
   5. Explain architecture tab (1 min)
   ```

3. **Have screenshots ready** as backup
   - In case of connectivity issues
   - Can walk through features offline

4. **Prepare your talking points**:
   - "This is hosted on [Platform] demonstrating cloud deployment"
   - "The system uses RAG to ground responses in real documents"
   - "Notice how it cites sources for transparency"
   - "The architecture is scalable to millions of documents"

### During Demo:

**Script Example:**
> "Let me show you a live demo. This is hosted on Streamlit Cloud and demonstrates the complete RAG pipeline. I'll load some sample documents about AI and technology topics... [click Load Sample Documents]... Now the system has processed these documents and created vector embeddings. Let me query it: 'What is machine learning?' ... [submit query]... As you can see, it retrieves relevant context and generates an accurate answer with source attribution. The sources shown here are the actual document chunks used to generate the response."

---

## 🔧 Troubleshooting Common Issues

### Issue: "Requirements too large"
**Solution:** Create a lighter `requirements.txt`:
```txt
sentence-transformers==2.3.1
faiss-cpu==1.7.4
streamlit==1.29.0
pypdf==3.17.4
python-dotenv==1.0.0
```

### Issue: "Memory limit exceeded"
**Solution:** 
- Use smaller embedding model in production
- Reduce number of sample documents
- Add memory limits in code

### Issue: "OpenAI API key errors"
**Solution:**
- The system works in "mock mode" without API key
- For interview demo, mock mode is actually fine
- Shows the system without costs

### Issue: "Slow cold start"
**Solution:**
- First load takes 30-60 seconds (downloading model)
- Subsequent loads are instant (cached)
- Mention this is expected in free tier

---

## 🎯 Recommended Approach for Interview

**Best Option: Streamlit Cloud**

1. Deploy to Streamlit Cloud (takes 5 minutes)
2. Test it works perfectly
3. Share URL in your resume/LinkedIn
4. During interview: "I have a live demo deployed at [URL]"
5. Walk through it showing key features
6. Explain architecture and technical decisions

**Backup Plan:**
- Have screenshots of the running app
- Have code open in GitHub
- Can run locally if needed with `streamlit run streamlit_app.py`

---

## 📊 What This Demonstrates to Interviewers

✅ **Cloud Deployment Skills** - You can ship production apps
✅ **Full-Stack Capability** - Backend + Frontend
✅ **Modern AI Tech** - RAG, embeddings, LLMs
✅ **User Experience** - Not just code, but usable product
✅ **Initiative** - Went beyond basic requirements
✅ **Presentation Skills** - Can demo your work effectively

---

## 🚀 Next Steps

1. **Deploy to Streamlit Cloud** (start here!)
2. **Test thoroughly** with different queries
3. **Share the link** on LinkedIn and resume
4. **Prepare demo script** for interview
5. **Practice talking through** the features

The live demo will make a huge impression! It shows you can not only build systems but also deploy and present them professionally.

Good luck! 🎉
