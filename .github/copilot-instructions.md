# AI Coding Agent Instructions - Letterboxd Roaster Backend

## Project Overview
Letterboxd Roaster is a **movie recommendation API** combining content-based filtering (cosine similarity) with collaborative filtering (SVD matrix factorization) to generate personalized movie recommendations. It integrates Gemini API for text generation and TMDB API for movie data.

## Architecture & Data Flow

### Recommendation Pipeline (Core Logic)
```
user_input (movie title) 
  ↓
get_candidates() → cosine similarity search (20 candidates)
  ↓
rerank_with_svd() → predict user preference scores
  ↓
top 5 recommendations + Gemini-generated roast
```

### Key Components
- **models.py**: Loads three pickle models at startup: `svd_model` (collaborative filtering), `cosine_sim` (content-based similarity matrix), `movies_df` (movie metadata)
- **main.py**: FastAPI app stub - extend with `/recommend` endpoint that calls `get_candidates()` and `rerank_with_svd()`
- **Persona Mapping**: Three hardcoded user IDs map to viewing styles (casual: 52960, enthusiast: 72315, critic: 80974)
- **Data Normalization**: `movies_df['title_lower']` enables case-insensitive title lookup

## Critical Patterns & Gotchas

### Model Loading Pattern
- All model files must exist in `models/` directory before importing (fatal runtime error if missing)
- Models are loaded at module import time in `models.py` - not lazily on-demand
- Paths are relative to current working directory; run uvicorn from project root

### Bug in models.py Line ~44
Variable typo: loop uses `tile` but appends `title` (undefined variable) - fix when extending `rerank_with_svd()`

### Persona System
Use persona IDs with SVD predictions instead of actual user IDs - persona_map provides consistent behavior for different user archetypes

### External APIs
- **Gemini API**: Called via `google-generativeai` - requires `GEMINI_API_KEY` in `.env`
- **TMDB API**: Optional currently; requires `TMDB_API_KEY` in `.env`
- Both configured via `python-dotenv` from `.env`

## Development Workflow

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY and TMDB_API_KEY to .env
```

### Running
```bash
uvicorn main:app --reload
```
Server runs on `http://localhost:8000`; use `/health` endpoint to verify startup.

### Adding Endpoints
1. Import model functions in `main.py`: `from models import get_candidates, rerank_with_svd`
2. Create FastAPI route that accepts movie title + optional persona
3. Call `get_candidates(title)` then `rerank_with_svd(candidates, persona_id)`
4. Use Gemini API to generate roast text (model: `gemini-pro` or latest)

## Code Style & Conventions
- **Python 3.9+** with type hints preferred for new functions
- Use lowercase snake_case for function/variable names (existing code follows this)
- Pandas operations on `movies_df`: avoid in-place modifications; return new DataFrames
- Error handling: raise `ValueError` for data issues (e.g., movie not found), let FastAPI handle HTTP responses
