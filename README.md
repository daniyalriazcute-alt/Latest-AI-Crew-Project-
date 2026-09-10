# 🎓 Learning Accelerator – Adaptive AI Tutor

A multi-agent AI tutor built with CrewAI, Streamlit, Groq, FAISS, and Sentence-Transformers. Runs on Google Colab, deploys to Streamlit Community Cloud, and works on the Groq free tier.

## Features

- **4 CrewAI agents**: Planner, Explainer (RAG), Quiz Generator, Progress Coach
- **RAG pipeline**: Upload PDF/DOCX/TXT/MD → chunked → embedded → FAISS indexed → retrieved for grounded answers
- **Free LLM**: Groq's `openai/gpt-oss-20b` on the free tier
- **Session memory**: Score history and "thought signatures" tracked per session
- **Prompt injection resistance**: Document text treated as untrusted reference material
- **No hard-coded API keys**: Uses Streamlit Secrets or environment variables

## Architecture

```
Student
  |
  v
Streamlit UI (4 tabs)
  |
  +--> Planner Agent ---------> Groq LLM
  |
  +--> Explainer Agent
  |       |
  |       +--> Sentence-Transformers (embeddings)
  |       +--> FAISS (retrieval)
  |       +--> Groq LLM
  |
  +--> Quiz Generator Agent --> Groq LLM
  |
  +--> Progress Coach Agent --> Session memory + Groq LLM
```

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit 1.50 |
| Agent framework | CrewAI 1.8.1 |
| LLM routing | LiteLLM 1.77 (patched for Groq) |
| LLM | Groq `openai/gpt-oss-20b` |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` |
| Vector search | FAISS (CPU) |
| Doc parsing | pypdf, python-docx |
| Runtime | Google Colab (free) |
| Tunnel | Cloudflare Tunnel (free) |

## Quick Start (Local)

```bash
git clone https://github.com/YOUR_USERNAME/learning-accelerator.git
cd learning-accelerator
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Run:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → select your repo → main file: `app.py`.
4. Open **Settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**.

## Run on Google Colab + Cloudflare Tunnel

See `COLAB_GUIDE.md` for the full walkthrough.

Quick version:

```python
!pip install -q streamlit "crewai[tools,litellm]==1.8.1" litellm==1.77.0 openai==1.99.5 sentence-transformers faiss-cpu pypdf python-docx numpy
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
```

Then write `app.py`, set `GROQ_API_KEY`, start Streamlit on port 8501, start `cloudflared tunnel --url http://localhost:8501`, and open the generated `trycloudflare.com` URL.

## The LiteLLM Patch

Groq's API rejects fields like `cache_breakpoint` and `is_litellm` that LiteLLM injects. The `app.py` patches `litellm.completion` to strip those fields before the request leaves the app. This is what makes CrewAI work with Groq's free tier.

## Security

- Secrets never in source code
- 10 MB per-file upload limit
- File-type whitelist (PDF, DOCX, TXT, MD)
- Document text treated as untrusted
- Prompt injection resistance in the RAG prompt
- Source attribution in every grounded answer

## Limitations

- Session-only memory (no database)
- No authentication
- Free tier rate limits apply (1,000 requests/day)

## License

MIT — see `LICENSE`.