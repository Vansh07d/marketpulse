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
