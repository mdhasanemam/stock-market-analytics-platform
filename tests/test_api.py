from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_health_and_pages(client):
    assert client.get("/api/health").json()["data_provider"] == "csv"
    for path in ["/", "/dashboard", "/predict", "/compare", "/watchlist", "/history"]:
        r = client.get(path)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"], path
    assert client.get("/static/css/app.css").status_code == 200
    assert client.get("/docs").status_code == 200       # Swagger UI


def test_search_quote_history_indicators(client):
    assert client.get("/api/stocks/search?q=micro").json()["results"][0]["symbol"] == "MSFT"
    q = client.get("/api/stocks/AAPL/quote").json()
    assert q["close"] > 0 and q["as_of"] == "2017-11-10"
    h = client.get("/api/stocks/AAPL/history?range=3M").json()
    assert len(h["dates"]) == len(h["close"]) == len(h["volume"]) > 40
    custom = client.get("/api/stocks/AAPL/history?start=2016-01-01&end=2016-03-31").json()
    assert custom["summary"]["start"] >= "2016-01-01" and custom["summary"]["end"] <= "2016-03-31"
    ind = client.get("/api/stocks/MSFT/indicators?range=1Y").json()
    n = len(ind["dates"])
    assert len(ind["rsi"]) == n and len(ind["macd"]["hist"]) == n and len(ind["bollinger"]["upper"]) == n
    assert all(v is not None for v in ind["sma50"])          # warmed up on full history, not just the slice


def test_error_handling(client):
    assert client.get("/api/stocks/NOPE/quote").status_code == 404
    assert client.get("/api/stocks/A$B/quote").status_code == 422
    assert client.get("/api/stocks/AAPL/history?range=9Y").status_code == 422
    assert client.get("/api/stocks/AAPL/history?start=2017-05-01&end=2017-01-01").status_code == 422
    assert client.post("/api/predict", json={"symbol": "AAPL", "horizon": 99}).status_code == 422
    assert client.post("/api/predict", json={"symbol": "AAPL", "model": "magic"}).status_code == 422
    assert client.post("/api/predict", json={"symbol": "bad symbol!"}).status_code == 422
    assert client.post("/api/predict", json={"symbol": "NOPE"}).status_code == 404
    assert client.get("/api/history/999").status_code == 404


def test_predict_separates_backtest_from_future_and_saves(client):
    r = client.post("/api/predict", json={"symbol": "AAPL", "model": "bayesian_ridge", "horizon": 5, "period": "5Y"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["backtest"]["kind"] == "out_of_sample" and j["forecast"]["kind"] == "future"
    assert j["forecast"]["dates"][0] > j["as_of"] > j["backtest"]["dates"][0]
    assert j["backtest"]["dates"][-1] <= j["as_of"]
    assert {"mae", "mse", "rmse", "r2", "mape", "directional_accuracy", "mae_improvement_vs_persistence", "mae_improvement_vs_drift"} <= set(j["metrics"])
    assert "not investment advice" in j["disclaimer"] and j["verdict"]
    hist = client.get("/api/history").json()["items"]
    assert len(hist) == 1 and hist[0]["symbol"] == "AAPL" and hist[0]["model"] == "bayesian_ridge"
    full = client.get(f"/api/history/{j['saved_id']}").json()
    assert full["forecast"]["path"] == j["forecast"]["path"] and full["config"]["horizon"] == 5
    csv = client.get(f"/api/history/{j['saved_id']}/export.csv")
    assert csv.status_code == 200 and "future_forecast" in csv.text
    assert client.delete(f"/api/history/{j['saved_id']}").status_code == 200
    assert client.get("/api/history").json()["items"] == []


def test_predict_save_false_does_not_persist(client):
    client.post("/api/predict", json={"symbol": "MSFT", "model": "persistence", "horizon": 1, "save": False})
    assert client.get("/api/history").json()["items"] == []


def test_compare_leaderboard_and_baselines(client):
    r = client.post("/api/compare", json={"symbol": "TSLA", "models": ["bayesian_ridge", "decision_tree"], "horizon": 5, "period": "5Y"})
    assert r.status_code == 200, r.text
    j = r.json()
    ids = [row["id"] for row in j["leaderboard"]]
    assert {"persistence", "drift", "bayesian_ridge", "decision_tree"} <= set(ids)   # baselines always included
    maes = [row["metrics"]["mae"] for row in j["leaderboard"]]
    assert maes == sorted(maes)                                                      # leaderboard is sorted
    assert all("beats_all_baselines" in row for row in j["leaderboard"])
    assert len(j["series"]["dates"]) == len(j["series"]["actual"]) == len(j["series"]["predicted"]["bayesian_ridge"])
    assert j["best_ml_model"] in {"bayesian_ridge", "decision_tree"}


def test_watchlist_crud(client):
    assert client.get("/api/watchlist").json()["items"] == []
    assert client.post("/api/watchlist", json={"symbol": "aapl"}).status_code == 201
    assert client.post("/api/watchlist", json={"symbol": "AAPL"}).status_code == 409
    assert client.post("/api/watchlist", json={"symbol": "NOPE"}).status_code == 404
    assert client.post("/api/watchlist", json={"symbol": "bad sym"}).status_code == 422
    items = client.get("/api/watchlist").json()["items"]
    assert [i["symbol"] for i in items] == ["AAPL"] and items[0]["change_pct"] is not None
    assert client.delete("/api/watchlist/AAPL").status_code == 200
    assert client.delete("/api/watchlist/AAPL").status_code == 404


def test_overview_survives_partial_failure(client):
    j = client.get("/api/overview").json()
    assert len(j["featured"]) == 4 and all(t["trend"] in {"Bullish", "Bearish", "Neutral"} for t in j["trend"])


def test_rate_limit(tmp_path):
    s = Settings(data_provider="csv", database_url="sqlite:///:memory:", cache_dir=tmp_path, rate_limit_per_minute=3)
    c = TestClient(create_app(s))
    codes = [c.get("/api/health").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
