# Security notes

**Reporting:** please open a private security advisory on GitHub (Security tab → "Report a vulnerability") rather than a public issue.

## What this project does
* No API keys are required (Yahoo Finance via `yfinance` needs none). If you add a keyed provider, put the key in `.env` / your host's secret store. `.env` is git-ignored and never copied into the Docker image.
* Inputs are validated (ticker regex, bounded horizon, enum periods/models, date parsing); errors return short messages, never stack traces.
* Rate limiting per IP, CORS limited to `CORS_ORIGINS`, security headers (`nosniff`, `X-Frame-Options: DENY`, referrer/permissions policy).
* With `ENVIRONMENT=production` the Swagger/OpenAPI endpoints are disabled.
* The Docker image runs as a non-root user.
* Tests check that headers are present, docs are hidden in production, `.env`/DB files are ignored and no secret-like strings are committed.

## Known gaps
* No authentication: the watchlist and prediction history are shared by everyone who can reach the server. **Do not expose a public instance with data you consider private**, or put it behind an authenticating proxy.
* In-memory rate limiting works per process only. Put a gateway/WAF in front for real traffic.
* No Content-Security-Policy yet (pages use inline scripts and CDN libraries; Bootstrap/Plotly are not pinned with SRI hashes).
* Dependencies are version-ranged, not locked; run `pip install pip-audit && pip-audit` periodically.
