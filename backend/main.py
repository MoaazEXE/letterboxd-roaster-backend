import asyncio
import logging
import re
from contextlib import asynccontextmanager

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .llm import RoastClient, get_roast_client, init_roast_client
from .recommender import (
    MovieNotFoundError,
    Recommender,
    get_recommender,
    init_recommender,
)
from .schemas import (
    PERSONA_TO_USER_ID,
    HealthResponse,
    Persona,
    Recommendation,
    RecommendResponse,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("letterboxd_roaster")

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def _clean_title_for_tmdb(raw: str) -> str:
    # Strip year: "Inception (2010)" → "Inception"
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", raw).strip()
    # Fix "Prestige, The" → "The Prestige"
    title = re.sub(r"^(.+),\s*(The|A|An)\s*$", r"\2 \1", title, flags=re.IGNORECASE)
    return title


async def fetch_poster_url(title: str, api_key: str | None) -> str | None:
    if not api_key:
        return None
    query = _clean_title_for_tmdb(title)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                TMDB_SEARCH_URL,
                params={"api_key": api_key, "query": query, "include_adult": "false"},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results and results[0].get("poster_path"):
                return f"{TMDB_IMAGE_BASE}{results[0]['poster_path']}"
    except Exception:
        logger.warning("TMDB lookup failed for: %s (cleaned: %s)", title, query)
    return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting up — loading models and clients")
    await run_in_threadpool(init_recommender)
    init_roast_client()
    logger.info("Startup complete")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Letterboxd Roaster API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(meta_router)
    app.include_router(recommend_router)
    return app


meta_router = APIRouter(tags=["meta"])
recommend_router = APIRouter(tags=["recommend"])


@meta_router.get("/", response_model=dict[str, str])
def root() -> dict[str, str]:
    return {"message": "Letterboxd Roaster backend is running"}


@meta_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@recommend_router.get("/recommend", response_model=RecommendResponse)
async def recommend(
    movie: str = Query(..., min_length=1, description="Input movie title"),
    persona: Persona = Query(..., description="One of: casual, enthusiast, critic"),
    recommender: Recommender = Depends(get_recommender),
    roast_client: RoastClient = Depends(get_roast_client),
) -> RecommendResponse:
    settings = get_settings()
    user_id = PERSONA_TO_USER_ID[persona]

    try:
        candidates = await run_in_threadpool(recommender.get_candidates, movie)
    except MovieNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if not candidates:
        raise HTTPException(status_code=404, detail="No candidates returned for the provided movie title")

    try:
        top5 = await run_in_threadpool(recommender.rerank_with_svd, candidates, user_id)
    except Exception as e:
        logger.exception("SVD rerank failed")
        raise HTTPException(status_code=500, detail="Error during re-ranking") from e

    roasts, poster_urls = await asyncio.gather(
        asyncio.gather(*(roast_client.generate_roast(movie, title, pred) for title, _, pred in top5)),
        asyncio.gather(*(fetch_poster_url(title, settings.tmdb_api_key) for title, _, _ in top5)),
    )

    recommendations = [
        Recommendation(
            title=title,
            movieId=movie_id,
            predicted_rating=round(pred, 2),
            roast=roast,
            poster_url=poster_url,
        )
        for (title, movie_id, pred), roast, poster_url in zip(top5, roasts, poster_urls)
    ]

    return RecommendResponse(
        input_movie=movie, persona=persona, recommendations=recommendations
    )


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
