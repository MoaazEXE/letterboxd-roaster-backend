import os
import asyncio
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Simple deterministic fallback roast generator for offline/dev use.
def _fallback_roast(input_movie: str, recommended_movie: str, predicted_rating: float) -> str:
    return (
        f"You liked '{input_movie}', so here's '{recommended_movie}' — rated {predicted_rating:.1f}. "
        "Expect moody cinematography, too many close-ups, and a runtime that thinks it's cleverer than you."
    )

async def generate_roast(input_movie: str, recommended_movie: str, predicted_rating: float) -> str:
    """
    Generate a 2-3 sentence snarky roast for a recommendation using Google Gemini.
    """
    if not client:
        return _fallback_roast(input_movie, recommended_movie, predicted_rating)

    prompt = (
        f"The user liked the movie '{input_movie}'. I am recommending them '{recommended_movie}' "
        f"with a predicted rating of {predicted_rating:.1f}/5.0. "
        f"Write a 2-3 sentence snarky, film-snob, Letterboxd-style roast of this recommendation/viewer. "
        f"Be witty, critical, and slightly pretentious. Do not use emojis."
    )

    try:
        # Using the async client (aio) for integration with the rest of the app
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return _fallback_roast(input_movie, recommended_movie, predicted_rating)