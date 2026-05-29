from typing import Literal

from pydantic import BaseModel, Field


Persona = Literal["casual", "enthusiast", "critic"]


PERSONA_TO_USER_ID: dict[Persona, int] = {
    "casual": 52960,
    "enthusiast": 72315,
    "critic": 80974,
}


class Recommendation(BaseModel):
    title: str
    movie_id: int = Field(..., alias="movieId")
    predicted_rating: float
    roast: str
    poster_url: str | None = None

    model_config = {"populate_by_name": True}


class RecommendResponse(BaseModel):
    input_movie: str
    persona: Persona
    recommendations: list[Recommendation]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
