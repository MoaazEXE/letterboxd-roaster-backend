import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Optional: GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.0")  # adjust if you have model name

# Simple deterministic fallback roast generator for offline/dev use.
def _fallback_roast(input_movie: str, recommended_movie: str, predicted_rating: float) -> str:
    return (
        f"You liked '{input_movie}', so here's '{recommended_movie}' — rated {predicted_rating:.1f}. "
        "Expect moody cinematography, too many close-ups, and a runtime that thinks it's cleverer than you."
    )

async def generate_roast(input_movie: str, recommended_movie: str, predicted_rating: float) -> str:
    """
    Generate a 2-3 sentence snarky roast for a recommendation.
    If GEMINI_API_KEY is set, replace the TODO block with a real API call to Gemini.
    """
    if not GEMINI_API_KEY:
        # quick synchronous fallback wrapped in coroutine
        return _fallback_roast(input_movie, recommended_movie, predicted_rating)

    # TODO: Integrate with Google Gemini API here.
    # Implementation notes:
    # - Use your preferred HTTP client (aiohttp, httpx) or Google's official client libraries.
    # - Keep the prompt concise; include input_movie, recommended_movie, and predicted_rating.
    # - Return a 2-3 sentence snarky, Letterboxd-style roast.
    #
    # Example prompt:
    # prompt = (
    #   f"You liked '{input_movie}'. Recommend and roast '{recommended_movie}' for that viewer. "
    #   f"Predicted rating: {predicted_rating:.1f}. Give 2-3 sentences, film-snob, witty and snarky tone."
    # )
    #
    # For now, return fallback while the team wires Gemini credentials & client.
    await asyncio.sleep(0.05)  # small async pause to simulate network latency
    return _fallback_roast(input_movie, recommended_movie, predicted_rating)