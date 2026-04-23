import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def run_test():
    response = client.get("/recommend?movie=Interstellar&persona=enthusiast")
    print(f"Status Code: {response.status_code}")
    import json
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    run_test()
