from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import asyncio
import logging

# load .env
load_dotenv()

# import project modules (models.py provided by teammate)
# models must provide: get_candidates(title) -> list[(title, movieId)]
# and rerank_with_svd(candidates, user_id) -> list[(title, movieId, predicted_rating)]
from backend import models
from backend import llm




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("letterboxd_roaster")

app = FastAPI(title="Letterboxd Roaster API")

# Allow frontend on localhost:3000 by default. Adjust ALLOWED_ORIGINS in .env if needed.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSONA_MAP = {
    "casual": 52960,
    "enthusiast": 72315,
    "critic": 80974,
}


@app.get("/recommend")
async def recommend(movie: str = Query(..., min_length=1), persona: str = Query(...)):
    persona = persona.lower()
    if persona not in PERSONA_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown persona '{persona}'. Valid: {list(PERSONA_MAP.keys())}")

    user_id = PERSONA_MAP[persona]

    # Step 1: get 20 candidates from content-based model
    try:
        candidates = models.get_candidates(movie)  # expected: list of (title, movieId)
    except Exception as e:
        logger.exception("Error in get_candidates")
        raise HTTPException(status_code=404, detail=str(e) or "Movie not found")

    if not candidates:
        raise HTTPException(status_code=404, detail="No candidates returned for the provided movie title")

    # Step 2: rerank with SVD -> top 5
    try:
        top5 = models.rerank_with_svd(candidates, user_id)  # expected: list of (title, movieId, predicted_rating)
    except Exception as e:
        logger.exception("Error in rerank_with_svd")
        raise HTTPException(status_code=500, detail="Error during re-ranking")

    # Step 3: generate roasts concurrently
    async def gen_for(rec):
        title, movie_id, pred = rec
        roast = await llm.generate_roast(input_movie=movie, recommended_movie=title, predicted_rating=pred)
        return {"title": title, "predicted_rating": round(float(pred), 2), "roast": roast}

    tasks = [gen_for(rec) for rec in top5]
    try:
        recommendations = await asyncio.gather(*tasks)
    except Exception:
        logger.exception("LLM generation error")
        # fallback: return results without roasts
        recommendations = [{"title": t[0], "predicted_rating": round(float(t[2]), 2), "roast": ""} for t in top5]

    return {"recommendations": recommendations}