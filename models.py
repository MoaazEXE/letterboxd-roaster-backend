import os
import re
import pickle
import pandas as pd


# load models at startup (once, at import time)
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def _load_pickle(filename: str):
    path = os.path.join(_MODEL_DIR, filename)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"Failed to load {filename}: file not found at {path}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {filename} from {path}: {e}"
        )

svd_model = None
cosine_sim = None
movies_df = None

_models_loaded = False

def _load_models():
    global svd_model, cosine_sim, movies_df, _models_loaded

    if not _models_loaded:
        print("Loading models...")
        svd_model = _load_pickle("svd_model.pkl")
        print("svd loaded")

        cosine_sim = _load_pickle("cosine_sim.pkl")
        print("cosine sim loaded")

        movies_df = _load_pickle("movies_df.pkl")
        print("movies df loaded")

        _models_loaded = True

# 2. persona → userId mapping
_persona_map = {
    "casual": 52960,
    "enthusiast": 72315,
    "critic": 80974,
}


def get_user_id(persona: str) -> int:
    """Return the userId for a persona string.

    Args:
        persona: One of "casual", "enthusiast", or "critic".

    Returns:
        The integer userId.

    Raises:
        ValueError: If persona is not recognised.
    """
    try:
        return _persona_map[persona]
    except KeyError:
        valid = ", ".join(_persona_map)
        raise ValueError(
            f"Invalid persona '{persona}'. Must be one of: {valid}"
        )



# 3. regex patterns to clean movie titles
_YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$") 
_ARTICLE_RE = re.compile(r",\s*the$", re.IGNORECASE) 
_LEADING_THE_RE = re.compile(r"^the\s+", re.IGNORECASE) 


def _normalise_title(title: str) -> str: #standardise movie title for matching
    t = title.lower().strip() #lowercase
    t = _YEAR_RE.sub("", t)     # "interstellar (2014)" → "interstellar"
    t = _ARTICLE_RE.sub("", t)   # "dark knight, the"    → "dark knight"
    t = _LEADING_THE_RE.sub("", t)  # "the dark knight"     → "dark knight"
    return t.strip()


_title_lookup = None

def _build_title_lookup():
    global _title_lookup

    if _title_lookup is None:
        _title_lookup = {}
        for _idx, _row in movies_df.iterrows():
            _norm = _normalise_title(_row["title"])
            if _norm not in _title_lookup:
                _title_lookup[_norm] = _idx



# 4. get_candidates
def get_candidates(title: str, n: int = 20) -> list[tuple[str, int, float]]:
    #return the top-n similar movies for a given title.
    # title: Movie title (case-insensitive, year and article optional).
    # n: Number of candidates to return (default 20).
    # return list of (title, movieId, similarity_score) tuples, sorted descending by similarity.
    _load_models()
    _build_title_lookup()

    if title is None:
        raise ValueError("Title cannot be None")
    if not isinstance(title, str):
        raise TypeError(f"Title must be a string, got {type(title).__name__}")
    if not title.strip():
        raise ValueError("Title cannot be empty")

    #normalise input title 
    norm = _normalise_title(title)

    #find movie index
    idx = _title_lookup.get(norm)

    #fallback: partial match (if exact match fail)
    if idx is None:
        for key in _title_lookup:
            if norm in key:
                idx = _title_lookup[key]
                break

    #if still not found, raise error
    if idx is None:
        raise ValueError("Movie not found. Please check the title and try again.")

    #get top similiar movies
    top_indices = cosine_sim["indices"][idx][:n]
    top_scores = cosine_sim["scores"][idx][:n]

    #build results list
    results = []
    for movie_idx, score in zip(top_indices, top_scores):
        movie = movies_df.iloc[movie_idx]
        results.append((movie["title"], int(movie["movieId"]), float(score)))

    return results


# ---------------------------------------------------------------------------
# 5. rerank_with_svd (personalisation)
# ---------------------------------------------------------------------------
def rerank_with_svd(
    candidates: list[tuple[str, int, float]], user_id: int
) -> list[tuple[str, int, float]]:
    #rerank candidates using SVD predicted ratings and return the top 5.
    # return Top 5 (title, movieId, predicted_rating) tuples sorted descending by predicted rating.

    results = []
    for item in candidates:
        title_str, movie_id = item[0], item[1]
        #predict rating for this user and movie using the SVD model
        pred = svd_model.predict(user_id, movie_id)

        results.append((title_str, int(movie_id), float(pred.est)))

    #sort by predicted rating (descending))
    results.sort(key=lambda x: x[2], reverse=True)

    return results[:5]


# ---------------------------------------------------------------------------
# 6. manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("models.py running")

    movie = "Interstellar"
    user_id = get_user_id("persona")

    #get similiar movies
    candidates = get_candidates(movie)

    if isinstance(candidates, str):
        print(candidates)  # error message
    else:
        print("Candidates (top 5):\n", candidates[:5])
        for i, (title, movie_id, score) in enumerate(candidates[:5], start=1):
            print(f"{i}. {title} | ID: {movie_id} | Similarity: {score:.3f}")

        #rerank using SVD
        print("\nTop 5 Recommendations:")
        top5 = rerank_with_svd(candidates, user_id)

        for i, (title, movie_id, rating) in enumerate(top5, start=1):
            print(f"{i}. {title} | ID: {movie_id} | Predicted Rating: {rating:.2f}")