import logging

from google import genai

from .config import get_settings


logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = (
    "The user liked the movie '{input_movie}'. I am recommending them "
    "'{recommended_movie}' with a predicted rating of {predicted_rating:.1f}/5.0. "
    "Write a 2-3 sentence snarky, film-snob, Letterboxd-style roast of this "
    "recommendation/viewer. Be witty, critical, and slightly pretentious. Do not use emojis."
)


def _fallback_roast(input_movie: str, recommended_movie: str, predicted_rating: float) -> str:
    return (
        f"You liked '{input_movie}', so here's '{recommended_movie}' — "
        f"rated {predicted_rating:.1f}. Expect moody cinematography, too many "
        "close-ups, and a runtime that thinks it's cleverer than you."
    )


class RoastClient:
    """Wraps the Gemini client with graceful fallback to a deterministic roast."""

    def __init__(self, api_key: str | None, model: str) -> None:
        self._model = model
        self._client = genai.Client(api_key=api_key) if api_key else None
        if self._client is None:
            logger.warning("GEMINI_API_KEY not set — roasts will use the offline fallback.")

    async def generate_roast(
        self, input_movie: str, recommended_movie: str, predicted_rating: float
    ) -> str:
        if self._client is None:
            return _fallback_roast(input_movie, recommended_movie, predicted_rating)

        prompt = _PROMPT_TEMPLATE.format(
            input_movie=input_movie,
            recommended_movie=recommended_movie,
            predicted_rating=predicted_rating,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model, contents=prompt
            )
            return (response.text or "").strip() or _fallback_roast(
                input_movie, recommended_movie, predicted_rating
            )
        except Exception as e:
            logger.warning("Gemini call failed, using fallback roast: %s", e)
            return _fallback_roast(input_movie, recommended_movie, predicted_rating)


_roast_client: RoastClient | None = None


def init_roast_client() -> RoastClient:
    global _roast_client
    if _roast_client is None:
        settings = get_settings()
        _roast_client = RoastClient(settings.gemini_api_key, settings.gemini_model)
    return _roast_client


def get_roast_client() -> RoastClient:
    if _roast_client is None:
        raise RuntimeError("Roast client not initialised. Call init_roast_client() first.")
    return _roast_client
