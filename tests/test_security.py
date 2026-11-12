import re
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_security_headers_on_pages_api_and_rate_limited_responses(tmp_path):
    s = Settings(data_provider="csv", database_url="sqlite:///:memory:", cache_dir=tmp_path, rate_limit_per_minute=1)
    c = TestClient(create_app(s))
    for path in ["/", "/api/health", "/api/health"]:       # the 2nd /api/health is a 429
        r = c.get(path)
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert "referrer-policy" in r.headers


def test_api_docs_hidden_in_production(tmp_path):
    s = Settings(data_provider="csv", database_url="sqlite:///:memory:", cache_dir=tmp_path, environment="production")
    c = TestClient(create_app(s))
    assert c.get("/docs").status_code == 404
    assert c.get("/openapi.json").status_code == 404
    assert c.get("/api/health").status_code == 200          # the API itself still works


def test_errors_do_not_leak_internals(client):
    r = client.get("/api/stocks/NOPE/quote")
    assert "Traceback" not in r.text and "site-packages" not in r.text


def test_env_files_and_databases_are_gitignored():
    ignore = (ROOT / ".gitignore").read_text().splitlines()
    for pattern in [".env", "*.db", "data/cache/"]:
        assert pattern in ignore


def test_repository_contains_no_secret_like_strings():
    pat = re.compile(r"(api[_-]?key|secret|token|passwd|password)\s*[:=]\s*['\"][A-Za-z0-9/+_\-]{12,}['\"]|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}", re.I)
    files = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in {".py", ".js", ".html", ".md", ".yml", ".yaml", ".txt", ".example", ".sh", ".json"}
             and not any(part in {".venv", ".git", "node_modules", "__pycache__", "sample"} for part in p.parts)]
    offenders = [str(p.relative_to(ROOT)) for p in files if p.name != "test_security.py" and pat.search(p.read_text(errors="ignore"))]
    assert offenders == []
