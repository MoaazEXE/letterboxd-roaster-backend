# Deployment Guide

This is a **two-part app**: a static React/Vite frontend and a Python/FastAPI ML
backend. They deploy to **different platforms**.

```
[ Browser ] ──> [ Frontend on Vercel ] ──HTTPS──> [ Backend on a container host ]
                  (static, free)                    (FastAPI + ML model, needs RAM)
```

## Why the backend can't go on Vercel

The recommender loads three pickled artefacts at startup:

| File             | Size    |
|------------------|---------|
| `svd_model.pkl`  | ~631 MB |
| `cosine_sim.pkl` | ~5 MB   |
| `movies_df.pkl`  | ~4 MB   |

`svd_model.pkl` alone is **~631 MB**. That:

- exceeds GitHub's 100 MB per-file limit (so it stays **gitignored** — never commit it),
- exceeds Vercel's serverless function bundle limit (250 MB),
- needs ~1 GB+ of RAM held in memory to serve requests,
- pulls in `scikit-surprise`, which compiles native extensions.

So the backend needs a **long-running container with real RAM**, not serverless.

## Recommended setup

### Frontend → Vercel
1. Import the **frontend** repo into Vercel. Framework preset: **Vite** (auto-detected).
2. Set env var `VITE_API_URL` = the deployed backend URL (e.g. `https://<your-backend-host>`).
3. Deploy. The included `vercel.json` rewrites all routes to `index.html` so client-side
   routing (e.g. refreshing `/results`) doesn't 404.

### Backend → a container host with ≥1 GB RAM
Good options (cheapest-first):

- **Hugging Face Spaces (Docker)** — free CPU tier with **16 GB RAM**. Ideal for an ML
  demo. Upload the `.pkl` files to the Space repo via Git LFS (or an HF model repo).
- **Railway** — usage-based, ~$5 free credit/month; allocate ~1 GB.
- **Fly.io** — configurable VM (e.g. `shared-cpu-1x` with 1 GB).
- **Render** — works, but the free/Starter tiers are 512 MB RAM and **will OOM** loading
  the 631 MB model. You'd need the 2 GB Standard plan (~$25/mo).

A `Dockerfile` is included. Build/run locally to verify:

```bash
docker build -t letterboxd-roaster-api .
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=... -e TMDB_API_KEY=... \
  -e ALLOWED_ORIGINS='["https://your-frontend.vercel.app"]' \
  -v "$(pwd)/backend/models:/app/backend/models" \
  letterboxd-roaster-api
```

### Getting the model onto the host
The image does **not** bake in the `.pkl` files. Pick one:

1. **Git LFS** — track `backend/models/*.pkl` with LFS on the deploy host (works well on
   HF Spaces).
2. **Object storage + startup download** — upload the files to S3 / Cloudflare R2 / the
   HF Hub and download them on first boot before the model loads.
3. **Mounted volume** — attach a persistent volume and upload the files once.

### Required environment variables (backend)
| Var               | Notes                                                            |
|-------------------|------------------------------------------------------------------|
| `GEMINI_API_KEY`  | Optional — without it, roasts use a deterministic fallback.      |
| `TMDB_API_KEY`    | Optional — without it, `poster_url` is `null`.                   |
| `ALLOWED_ORIGINS` | **Set to your Vercel frontend URL** as a JSON array.             |
| `PORT`            | Injected by most hosts; the container respects it.               |

> **Tip:** to make any platform fit a free tier, shrink `svd_model.pkl` by retraining the
> SVD with fewer latent factors and dumping with compression. Sub-100 MB opens up far
> more (and cheaper) hosting options.
