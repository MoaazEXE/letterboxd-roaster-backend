# Letterboxd Roaster

A fun web app that roasts your Letterboxd movie taste.

## Project Structure

- `backend/` - FastAPI backend with recommendation logic and AI roasts.
- `src/` - React frontend.
- `public/` - Static assets.

## Tech Stack

### Frontend
- React
- Vite

### Backend
- **FastAPI** - Web framework
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **Surprise** - Recommendation algorithms
- **Gemini API** - AI text generation

## Getting Started

### Frontend

1. Install dependencies:
   ```bash
   npm install
   ```
2. Run development server:
   ```bash
   npm run dev
   ```

### Backend

1. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Run development server:
   ```bash
   python -m uvicorn backend.main:app --reload
   ```
