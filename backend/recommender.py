import logging
import pickle
import re
from pathlib import Path

from .config import get_settings


logger = logging.getLogger(__name__)


_YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$")
_TRAILING_ARTICLE_RE = re.compile(r",\s*the$", re.IGNORECASE)
_LEADING_THE_RE = re.compile(r"^the\s+", re.IGNORECASE)


class MovieNotFoundError(Exception):
    """Raised when an input title cannot be matched against the catalogue."""


class Recommender:
    """Loads the SVD + cosine-similarity artefacts and serves recommendations."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._svd_model = None
        self._cosine_sim = None
        self._movies_df = None
        self._title_lookup: dict[str, int] = {}

    def load(self) -> None:
        logger.info("Loading recommender artefacts from %s", self._model_dir)
        self._svd_model = self._load_pickle("svd_model.pkl")
        self._cosine_sim = self._load_pickle("cosine_sim.pkl")
        self._movies_df = self._load_pickle("movies_df.pkl")
        self._build_title_lookup()
        logger.info("Recommender ready (%d titles indexed)", len(self._title_lookup))

    def _load_pickle(self, filename: str):
        path = self._model_dir / filename
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Required model file missing: {path}. "
                "See README for where to obtain the .pkl artefacts."
            ) from e

    def _build_title_lookup(self) -> None:
        lookup: dict[str, int] = {}
        for idx, row in self._movies_df.iterrows():
            norm = self._normalise_title(row["title"])
            lookup.setdefault(norm, idx)
        self._title_lookup = lookup

    @staticmethod
    def _normalise_title(title: str) -> str:
        t = title.lower().strip()
        t = _YEAR_RE.sub("", t)
        t = _TRAILING_ARTICLE_RE.sub("", t)
        t = _LEADING_THE_RE.sub("", t)
        return t.strip()

    def get_candidates(self, title: str, n: int = 20) -> list[tuple[str, int, float]]:
        if not title or not title.strip():
            raise MovieNotFoundError("Title cannot be empty")

        norm = self._normalise_title(title)
        idx = self._title_lookup.get(norm)

        if idx is None:
            for key, candidate_idx in self._title_lookup.items():
                if norm in key:
                    idx = candidate_idx
                    break

        if idx is None:
            raise MovieNotFoundError(
                f"Movie '{title}' not found in catalogue."
            )

        top_indices = self._cosine_sim["indices"][idx][:n]
        top_scores = self._cosine_sim["scores"][idx][:n]

        results: list[tuple[str, int, float]] = []
        for movie_idx, score in zip(top_indices, top_scores):
            row = self._movies_df.iloc[movie_idx]
            results.append((row["title"], int(row["movieId"]), float(score)))
        return results

    def rerank_with_svd(
        self, candidates: list[tuple[str, int, float]], user_id: int, top_k: int = 5
    ) -> list[tuple[str, int, float]]:
        results = [
            (title, movie_id, float(self._svd_model.predict(user_id, movie_id).est))
            for title, movie_id, _score in candidates
        ]
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]


_recommender: Recommender | None = None


def init_recommender() -> Recommender:
    global _recommender
    if _recommender is None:
        settings = get_settings()
        _recommender = Recommender(settings.model_dir)
        _recommender.load()
    return _recommender


def get_recommender() -> Recommender:
    if _recommender is None:
        raise RuntimeError("Recommender not initialised. Call init_recommender() first.")
    return _recommender
