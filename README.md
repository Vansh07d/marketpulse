# MarketPulse

MarketPulse is a polling-based financial-news sentiment prototype. It fetches news and market quotes, scores new headlines with FinBERT, generates heuristic BUY/HOLD/SELL signals, and stores the latest state plus bounded signal history in Redis. These signals are experimental and are **not investment advice**.

## Requirements

- Python 3.12
- Redis 7 (local) or Docker Compose
- API keys for at least one supported news/market provider
- Your trained FinBERT model files, unless you configure a Hugging Face model ID

## Windows / PowerShell setup

Run from the repository root (the folder containing `requirements.txt`):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch
Copy-Item .env.example .env
```

Edit `.env` and add whichever provider API keys you have. `torch` is installed separately so Docker can use the CPU-only wheel while Windows can choose an appropriate wheel for your machine. If the PyTorch download times out, retry with:

```powershell
python -m pip install torch --timeout 300 --retries 10
```

Run the model inference smoke test:

```powershell
python sentiment_service.py
```

Then run the worker and dashboard in separate activated terminals:

```powershell
python realtime_worker.py
```

```powershell
streamlit run app2.py
```

For the first test, keep `DEFAULT_SYMBOL=INFY`. The worker polls at `POLL_INTERVAL_SECONDS` (default 60 seconds); set this no lower than 10 seconds to avoid excessive provider requests.

## Trained model layout

The default model path is `models/finbert_model`. Supported local layouts:

```text
models/finbert_model/
  config.json
  model.safetensors (or pytorch_model.bin)
  tokenizer.json / tokenizer_config.json / vocab files
```

or the two-fold training output:

```text
models/finbert_model/
  tokenizer.json / tokenizer_config.json / vocab files
  fold_1/
    config.json
    model.safetensors (or pytorch_model.bin)
  fold_2/
    config.json
    model.safetensors (or pytorch_model.bin)
```

Set `MODEL_PATH` if your model is elsewhere. Alternatively, set `MODEL_ID` to a Hugging Face model repository; for a private repo, set `HF_TOKEN` in `.env`. Never commit `.env` or access tokens. The ZIP does not include model weights; copy them separately or configure `MODEL_ID`.

## Docker Compose

1. Copy `.env.example` to `.env` and add your API keys.
2. Put model files under `models/finbert_model` or configure `MODEL_ID`.
3. Run:

```powershell
docker compose up --build
```

4. Open <http://localhost:8503>. Redis is published on host port `6390`; containers connect to it at `redis:6379`.

The Docker image installs CPU-only PyTorch to avoid downloading CUDA libraries. The Docker build still needs internet access for Python packages and any Hugging Face model not supplied locally. The model folder is not ignored by `.dockerignore`; large model weights can make the image large.

## Useful checks

```powershell
python -m compileall .
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

To run the signal engine's local smoke test:

```powershell
python signal_service.py
```

## Current limitations

- News/quote updates are polling-based, not a push streaming feed.
- The worker currently processes one configured symbol per process.
- Signal weights and thresholds are heuristic and require backtesting.
- Signal history is bounded to the most recent 1,000 snapshots and expires after 30 days.
- Provider availability, API quotas, and symbol coverage depend on the configured vendors.

## Architecture: two paths, one is primary

| Path | Files | Status |
|---|---|---|
| **Streaming (primary)** | `streaming/ingestor.py` -> `news.raw` -> `streaming/consumer.py` (FinBERT, micro-batches) -> `news.scored` -> `streaming/aggregator.py` -> Redis signals (dashboard) + SQLite | Use this |
| Polling worker (legacy) | `realtime_worker.py` | Single symbol, kept for quick demos; do not run it together with the aggregator for the same symbol (both write the same Redis keys) |

```powershell
docker compose --profile stream up -d --build redis ingestor inference aggregator metrics dashboard
# (do NOT use a bare `up -d`: it also starts the legacy `worker`, which fights the aggregator over the same keys)
python verify_stack.py --count 100 --rate 5       # replay events carry origin=replay, so they never reach the dashboard or SQLite
```

Measured on CPU with the 2-fold FinBERT ensemble: p95 end-to-end 1.2 s at 5 events/s; about 6 events/s sustained capacity.
That is **scoring** latency. News freshness is bounded by `INGEST_INTERVAL_SECONDS` (default 300) plus the provider's own delay.

## Research workflow (is the signal any good?)

Every scored headline and every signal is kept in SQLite (`MARKETPULSE_DB`, a Docker volume), not only in Redis.

```powershell
# 1. After a few days of running: label real headlines and measure the model on them
python tools/eval_headlines.py --export 200 --out eval/headlines.csv     # fill the `label` column by hand
python tools/eval_headlines.py --eval eval/headlines.csv

# 2. After 2+ weeks of signals: backtest vs forward returns (next-day-close entry, costs, time-based split)
python backtest.py --db data/marketpulse.db --yf --suffix .NS --horizon 1 --cost-bps 20
```

Weights and thresholds in `signal_service.py` are heuristics. Tune them only on the "tune" split; look at "test" once.

## Fine-tuning (only if needed)

The current model is ProsusAI/finbert further fine-tuned on `all-data.csv` (Financial PhraseBank, 50%-agreement). Measure first
(`tools/eval_headlines.py`). If accuracy on your own real headlines is below about 80%, add domain data (CSV with `text,label`):

```powershell
$env:EXTRA_DATA = "data_extra/indian_headlines.csv,data_extra/fiqa.csv"; python finbert.py
```

Extra rows are added to training only; the held-out PhraseBank test split is unchanged so results stay comparable.

## Operations: metrics and failure drills

- `http://localhost:9108/metrics` (Prometheus text) and `/healthz`: stream lengths, group pending/lag, p50/p95/p99 per hop,
  dead-letter reasons, heartbeat ages. Alert ideas: `mp_group_lag > 100`, `mp_latency_ms{hop="e2e_ms",quantile="0.95"} > 2000`,
  `mp_heartbeat_age_seconds > 60`, `mp_stream_length{stream="dead"}` rising.
- `python tools/drill.py` kills the consumer, the aggregator and Redis in turn and checks nothing is lost or duplicated.
  Results are written to `drill_results.md`.

## Measured results (CPU, 2-fold FinBERT ensemble, full stack running)

| Metric | Value |
|---|---|
| End-to-end scoring latency at 5 events/s | p50 789 ms, p95 1424 ms, p99 1491 ms (mean batch 3.1) |
| Sustained capacity | about 6 events/s (20 events/s builds a queue: p95 12.2 s) |
| Held-out accuracy (PhraseBank, 484 sentences) | 88.6%, macro-F1 0.877 |
| Failure drills | see `drill_results.md` |

Latency is queue + inference inside this pipeline. News freshness is additionally bounded by the poll interval and the provider's delay.