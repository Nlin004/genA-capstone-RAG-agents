## Structure

- **Manager Agent** (`agents/manager_agent.py`) — classifies the query,
  routes to one or both sub-agents, merges labeled results, or asks a
  clarifying question if the query is ambiguous.
- **Qualitative RAG Agent** (`agents/qualitative_agent.py`) — embeds the
  query with `sentence-transformers`, retrieves the top-k chunks from a
  persistent Chroma collection, drops anything below the relevance
  threshold, and asks Gemini to answer using only the retrieved chunks.
  Returns citations with doc id + similarity score.
- **Quantitative NL-to-SQL Agent** (`agents/quantitative_agent.py`) —
  sends the DB schema + question to Gemini, validates the returned SQL is
  a single read-only `SELECT`, executes it against SQLite, optionally
  renders a chart, and asks Gemini to summarize the result.

## Setup Instructions:


### Initial Environment
- cd to `rag_capstone/` as root level
- Activate venv 
- `python3.12 -m pip install -r requirements.txt`
- Create real env: `cp .env.example .env`
- Build dummy SQLite database `python3.12 -m scripts.setup_dummy_db`
- Ingest dummy docs into chroma: `python3.12 -m scripts.ingest_docs`

### Run Automated Test Suites:
`pytest -q`

## Running the CLI

```bash
python cli.py
```

Type a question, or `exit`/`quit` to leave.

## Running the API

```bash
uvicorn api.main:app --reload
```

- Interactive docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- `POST /query` — routed through the Manager Agent
- `POST /qualitative` — Qualitative Agent only
- `POST /quantitative` — Quantitative Agent only

## Sample queries

**Qualitative:**
- "What is our company's security policy?"
- "Explain the code review process"
- "How do we handle customer complaints?"

**Quantitative:**
- "How many customers do we have per region?"
- "What's our customer churn rate?"
- "Compare invoice totals across regions"

**Complex / multi-agent:**
- "How does our employee satisfaction compare to our customer count, and what policies apply?"
