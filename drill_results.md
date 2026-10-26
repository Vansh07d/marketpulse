# Failure drills (2026-10-26 20:35 UTC)

| Drill | Result | Detail |
|---|---|---|
| inference killed mid-replay (300 events at 10/s) | PASS | all checks passed; zero loss/duplicates/dead; finished 49s after the kill; p50=15004 p95=22120 p99=47652 ms, mean batch 28.7 |
| aggregator killed mid-batch (50 events) | PASS | rows=50 distinct=50 (want 50); pending=0 (want 0) |
| redis restarted | PASS | scored stream 862 -> 862 entries (AOF kept data); services running: aggregator, dashboard, inference, ingestor, metrics, redis; post-restart replay passed (p50=392 p95=1265 p99=1379 ms, mean batch 2.2) |
