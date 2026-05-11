import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from google import genai

from backend.config import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY not set in .env")
    client = genai.Client(api_key=settings.gemini_api_key)
    for m in client.models.list():
        print(m.name)


if __name__ == "__main__":
    main()
