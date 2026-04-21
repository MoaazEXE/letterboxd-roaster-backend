import os
import re
import pickle
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Load models at startup (once, at import time)
# ---------------------------------------------------------------------------
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

svd_model = _load_pickle("svd_model.pkl")
cosine_sim = _load_pickle("cosine_sim.pkl")
movies_df = _load_pickle("movies_df.pkl")

# ---------------------------------------------------------------------------
# 2. Persona → userId mapping
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3. Build normalised title → DataFrame-index lookup (O(1) per query)
# ---------------------------------------------------------------------------
_YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$")
_ARTICLE_RE = re.compile(r",\s*the$", re.IGNORECASE)
_LEADING_THE_RE = re.compile(r"^the\s+", re.IGNORECASE)


def _normalise_title(title: str) -> str:
    """Lowercase, strip trailing year, strip leading/trailing article 'The'."""
    t = title.lower().strip()
    t = _YEAR_RE.sub("", t)           # "interstellar (2014)" → "interstellar"
    t = _ARTICLE_RE.sub("", t)        # "dark knight, the"    → "dark knight"
    t = _LEADING_THE_RE.sub("", t)    # "the dark knight"     → "dark knight"
    return t.strip()


# Map normalised title → df row index
_title_lookup: dict[str, int] = {}
for _idx, _row in movies_df.iterrows():
    _norm = _normalise_title(_row["title"])
    # first entry wins (duplicates are rare; keeps deterministic behaviour)
    if _norm not in _title_lookup:
        _title_lookup[_norm] = _idx


# ---------------------------------------------------------------------------
# 4. get_candidates
# ---------------------------------------------------------------------------
def get_candidates(title: str, n: int = 20) -> list[tuple[str, int, float]]:
    """Return the top-n similar movies for a given title.

    Args:
        title: Movie title (case-insensitive, year and article optional).
        n: Number of candidates to return (default 20).

    Returns:
        List of (title, movieId, similarity_score) tuples, sorted descending
        by similarity.

    Raises:
        TypeError: If title is not a string.
        ValueError: If title is None, empty, or not found in the database.
    """
    if title is None:
        raise ValueError("Title cannot be None")
    if not isinstance(title, str):
        raise TypeError(f"Title must be a string, got {type(title).__name__}")
    if not title.strip():
        raise ValueError("Title cannot be empty")

    norm = _normalise_title(title)

    idx = _title_lookup.get(norm)
    if idx is None:
        raise ValueError(f"Movie '{title}' not found in database")

    top_indices = cosine_sim["indices"][idx][:n]
    top_scores = cosine_sim["scores"][idx][:n]

    results = []
    for movie_idx, score in zip(top_indices, top_scores):
        movie = movies_df.iloc[movie_idx]
        results.append((movie["title"], int(movie["movieId"]), float(score)))

    return results


# ---------------------------------------------------------------------------
# 5. rerank_with_svd
# ---------------------------------------------------------------------------
def rerank_with_svd(
    candidates: list[tuple[str, int, float]], user_id: int
) -> list[tuple[str, int, float]]:
    """Rerank candidates using SVD predicted ratings and return the top 5.

    Args:
        candidates: Output of get_candidates — list of (title, movieId, score).
        user_id: Integer userId (obtain via get_user_id).

    Returns:
        Top 5 (title, movieId, predicted_rating) tuples sorted descending by
        predicted rating.
    """
    results = []
    for item in candidates:
        title_str, movie_id = item[0], item[1]
        pred = svd_model.predict(user_id, movie_id)
        results.append((title_str, int(movie_id), float(pred.est)))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:5]


# ---------------------------------------------------------------------------
# 6. Manual / ad-hoc testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("models.py running")

    movie = "Interstellar"
    user_id = get_user_id("enthusiast")

    candidates = get_candidates(movie)
    print("Candidates (top 5):", candidates[:5])

    top5 = rerank_with_svd(candidates, user_id)
    print("Top 5:", top5)
