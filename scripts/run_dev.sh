#!/usr/bin/env bash
cd "$(dirname "$0")/.."
exec python -m uvicorn backend.app.main:app --reload --port 8000
