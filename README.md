# Letterboxd Roaster (Backend)

Backend API for the Letterboxd Roaster project.

## Tech Stack

- **FastAPI** - Web framework
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **Surprise** - Recommendation algorithms
- **Gemini API** - AI text generation
- **TMDB API** - Movie data

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
uvicorn main:app --reload
```

## Project Structure

- `main.py` - FastAPI application entry point
- `models/` - Trained `.pkl` model files go here
- `.env` - Environment variables (create from `.env.example`)
