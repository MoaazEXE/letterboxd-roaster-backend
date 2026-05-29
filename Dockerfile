# syntax=docker/dockerfile:1
# Production image for the Letterboxd Roaster FastAPI backend.
FROM python:3.11-slim

# scikit-surprise compiles C/Cython extensions at install time, so we need a
# toolchain and numpy present during the build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install numpy first so scikit-surprise can build against it, then the rest.
COPY requirements.txt ./
RUN pip install --no-cache-dir "numpy<2" \
    && pip install --no-cache-dir -r requirements.txt

# Application code. NOTE: backend/models/*.pkl are NOT copied here — they are
# gitignored and far too large for an image layer (svd_model.pkl is ~631 MB).
# Provide them at runtime via a mounted volume or a startup download.
# See DEPLOYMENT.md.
COPY backend ./backend

ENV HOST=0.0.0.0
EXPOSE 8000

# Most hosts inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
