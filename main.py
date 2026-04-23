from fastapi import FastAPI
from models import get_candidates, rerank_with_svd, get_user_id

app = FastAPI(title="Letterboxd Roaster Backend")


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def root ():
    return {"message": "backend is running"}

#recommendation endpoint 
#connects all backend logic: 
# - takes movie title and persona as input,
# - finds similar movies using content-based filtering,
# - reranks them using SVD (personalisation),
# - returns the top 5 recommendations 
@app.get("/recommend")
def recommend(title: str, persona: str):
    try:
        user_id = get_user_id(persona) #convert persona to user_id
        candidates = get_candidates(title) #get similiar movies using cosine similarity
        top5 = rerank_with_svd(candidates, user_id) #rerank candidates using SVD predicted ratings 

        return { #format response as JSON
            "input_movie": title,
            "persona": persona,
            "recommendations": [
                {
                    "title": t,
                    "movieId": mid,
                    "predicted_rating": round(score, 2)
                }
                for t, mid, score in top5
            ]
        }

    except Exception as e: #handle errors (movie not found, invalid persona)
        return {"error": str(e)}
n