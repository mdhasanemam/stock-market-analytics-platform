# Deployment guide

> None of the platforms below were deployed to from this environment. The commands are standard but **unverified** — test on your account.
> The Docker image was **not built here** (no Docker daemon available); the CI workflow has not run on GitHub yet.

## Production start command
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```
Run **one worker**: the rate limiter and the model-result cache are in-process. For several workers/instances, move rate limiting to a gateway and caching to Redis.

## Environment variables
See `.env.example`. Set at least `DATA_PROVIDER`, `DATABASE_URL`, `CORS_ORIGINS`. No API key is required for yfinance; if you add a keyed provider, keep the key in the platform's secret store, never in the repo or frontend.

## Docker
```bash
cp .env.example .env
docker compose up --build        # http://localhost:8000
```
SQLite lives in the `appdata` volume. For PostgreSQL set `DATABASE_URL=postgresql+psycopg://…` and add `psycopg[binary]` to `requirements.txt`.

## Platform notes
| Platform | How | Caveats |
|---|---|---|
| **Render** | New Web Service → Docker (or Python: build `pip install -r requirements.txt`, start command above) | Free instances sleep after inactivity and have an ephemeral disk → SQLite history/watchlist is lost on redeploy; attach a disk or a Postgres instance. Cold starts delay the first request. |
| **Railway** | Deploy from GitHub; uses the Dockerfile | Free/trial credits are limited; add a volume or Postgres plugin for persistence. |
| **Hugging Face Spaces** | Docker Space; set `app_port: 8000` in the Space README front-matter | Storage is ephemeral unless persistent storage is enabled. |
| **AWS / GCP** | Container on App Runner / ECS / Cloud Run | Cloud Run is stateless → use Cloud SQL/Postgres. Set min instances ≥ 1 or expect cold starts. |

## Data-provider caveat for any cloud host
Yahoo Finance often throttles or blocks shared cloud IP ranges. The app retries, serves stale cache with a visible warning,
and returns HTTP 503 with a clear message — but if you need reliability in production, add a licensed provider
(subclass `DataProvider` in `backend/app/services/providers.py`) and use that instead.

## Compute caveat
A single-model forecast takes a few seconds; comparing all models takes roughly 10–60 s on a small instance (measured ~13–18 s per
ticker on the sandbox used for development, 9 models). Results are cached in memory. For heavy traffic, move `/api/predict` and
`/api/compare` to a background worker queue (Celery/RQ) — not implemented here.
