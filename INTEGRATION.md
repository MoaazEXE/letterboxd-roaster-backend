# Frontend ↔ Backend Integration

## Backend (already done)
- Listens on `http://127.0.0.1:8000`
- CORS allows `http://localhost:5173` (Vite default)
- Endpoint: `GET /recommend?movie={title}&persona={casual|enthusiast|critic}`

Run it: `python -m uvicorn backend.main:app --reload`

## Frontend (what to add)

### 1. Add a `.env` in the project root
```env
VITE_API_URL=http://127.0.0.1:8000
```

### 2. Call the API from React
```jsx
const res = await fetch(
  `${import.meta.env.VITE_API_URL}/recommend?movie=${encodeURIComponent(movie)}&persona=${persona}`
);
const data = await res.json();
// data.recommendations: [{ title, movieId, predicted_rating, roast }, ...]
```

## Response shape
```json
{
  "input_movie": "Inception",
  "persona": "casual",
  "recommendations": [
    { "title": "...", "movieId": 123, "predicted_rating": 4.32, "roast": "..." }
  ]
}
```

## Errors
- `404` — movie not in catalogue
- `422` — invalid persona
- `500` — SVD rerank failure
