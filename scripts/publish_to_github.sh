#!/usr/bin/env bash
# Creates the NEW public repo and pushes this project. Requires the GitHub CLI authenticated as the owner:  gh auth login
# The original mdhasanemam/stock-price-prediction repository is never touched.
set -euo pipefail
OWNER="${1:-mdhasanemam}"; REPO="stock-market-analytics-platform"
cd "$(dirname "$0")/.."
[ -d .git ] || { git init -b main && git add -A && git commit -m "Initial commit: stock market analytics platform"; }
gh repo create "$OWNER/$REPO" --public --source=. --remote=origin --push \
  --description "Stock analytics dashboard with walk-forward validated ML forecasts (FastAPI + Plotly)"
echo "Created https://github.com/$OWNER/$REPO"
