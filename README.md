# Letterboxd Roaster

A web app that recommends movies and roasts your taste, Letterboxd-style.

The backend pairs a content-based recommender (cosine similarity over movie features) with an SVD collaborative-filter rerank, then asks Gemini to write a snarky 2-3 sentence roast for each pick.

## Project Structure

```
backend/
  main.py          FastAPI app, lifespan, routes
  config.py        Pydantic Settings (loads .env)
  schemas.py       Request/response models, persona enum
  recommender.py   ML pipeline (load pkls, candidates, SVD rerank)
  llm.py           Gemini client with offline fallback
  models/          .pkl artefacts (gitignored — see below)
src/               React + Vite frontend
scripts/           Standalone dev/debug scripts
requirements.txt   Backend Python dependencies
.env.template      Copy to .env and fill in
```

## Prerequisites

- **Python 3.11 or 3.12** (3.13 has no prebuilt `scikit-surprise` wheel on Windows)
- **Node.js 18+** for the frontend
- The three `.pkl` model artefacts placed in [backend/models/](backend/models/):
  - `svd_model.pkl`
  - `cosine_sim.pkl`
  - `movies_df.pkl`

These are gitignored because they're large binary files; obtain them from the project owner or generate them with the training notebook.

## Backend setup

From the project root:

```powershell
# 1. Create venv (use Python 3.12 explicitly on Windows if 3.13 is your default)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.template .env
# Then edit .env and set GEMINI_API_KEY (optional — without it, roasts use a deterministic fallback)

# 4. Run the dev server
python -m uvicorn backend.main:app --reload
```

The API is now at <http://127.0.0.1:8000>. Interactive docs: <http://127.0.0.1:8000/docs>.

### Endpoints

| Method | Path         | Description                                  |
| ------ | ------------ | -------------------------------------------- |
| GET    | `/`          | Liveness check                               |
| GET    | `/health`    | Health check                                 |
| GET    | `/recommend` | `?movie=Inception&persona=casual` — returns top 5 with roasts |

`persona` must be one of `casual`, `enthusiast`, `critic`.

### Quick test

```powershell
python scripts\test_recommend.py
python scripts\test_llm.py
```

## Frontend setup

```powershell
npm install
npm run dev
```

The dev server runs on <http://localhost:5173> by default and is already in the backend's CORS allow-list.

## Configuration

All settings live in `.env` (see [.env.template](.env.template) for the full list):

- `GEMINI_API_KEY` — Google AI Studio key. Optional; without it, roasts use the offline fallback.
- `GEMINI_MODEL` — defaults to `gemini-flash-latest`.
- `HOST` / `PORT` — server bind address. Defaults to `127.0.0.1:8000`.
- `ALLOWED_ORIGINS` — JSON array of CORS origins.

## Tech Stack

**Backend:** FastAPI · Pydantic Settings · pandas · scikit-learn · scikit-surprise · google-genai
**Frontend:** React 19 · Vite
