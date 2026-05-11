import asyncio
import logging
from contextlib import asynccontextmanager

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

    roasts = await asyncio.gather(
        *(roast_client.generate_roast(movie, title, pred) for title, _, pred in top5)
    )

    recommendations = [
        Recommendation(
            title=title,
            movieId=movie_id,
            predicted_rating=round(pred, 2),
            roast=roast,
        )
        for (title, movie_id, pred), roast in zip(top5, roasts)
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
