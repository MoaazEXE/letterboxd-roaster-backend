from fastapi import FastAPI

app = FastAPI(title="Letterboxd Roaster Backend")


@app.get("/health")
def health_check():
    return {"status": "ok"}
