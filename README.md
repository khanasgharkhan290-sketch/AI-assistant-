
# 📄 ATS Resume Checker

A Streamlit app that analyzes your resume and returns:

- An **estimated ATS score** (70% Gemini AI analysis + 30% rule-based checks)
- **Prioritized improvements** with specific fixes
- **Strengths & weaknesses**
- **Keyword analysis**, optionally matched against a job description
- **Before/after rewrite examples**
- A downloadable JSON report

> ⚠️ The score is an *estimate*. Real ATS software varies by vendor, so treat this as guidance, not a guarantee.

## Tech stack

- [Streamlit](https://streamlit.io) – UI
- [Google Gemini Flash](https://ai.google.dev) via the `google-genai` SDK – analysis
- `pypdf` / `python-docx` – resume parsing (PDF, DOCX, TXT)

## Run locally

```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey), then create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
# Optional: change the model
# GEMINI_MODEL = "gemini-2.5-flash"
```

Start the app:

```bash
streamlit run app.py
```

If you skip the secrets file, the app will ask for the API key in the sidebar.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (never commit `secrets.toml`; it's in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Notes & limitations

- Scanned/image-only PDFs can't be read (and most ATS systems can't read them either). Use a text-based PDF or DOCX.
- Resume text is sent to Google's Gemini API. Don't upload anything you aren't comfortable sharing with that service.
- Max upload size: 5 MB.
