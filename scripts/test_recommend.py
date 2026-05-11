import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.main import app


def main() -> None:
    with TestClient(app) as client:
        response = client.get("/recommend", params={"movie": "Interstellar", "persona": "enthusiast"})
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
