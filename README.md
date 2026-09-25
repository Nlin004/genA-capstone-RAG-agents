# Initial Commit

## STEPS TO RUN:


### SETUP
- cd to `rag_capstone/` as root level
- Activate venv 
- `python3.12 -m pip install -r requirements.txt`
- Create real env: `cp .env.example .env`
- Build dummy SQLite database `python3.12 -m scripts.setup_dummy_db`
- Ingest dummy docs into chroma: `python3.12 -m scripts.ingest_docs`

### Run Automated Test Suites:
`pytest -q`