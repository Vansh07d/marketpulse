# Financial Sentiment Analysis Dashboard (FinBERT + Streamlit)

## Overview

This project is a **financial news sentiment analysis dashboard** that uses the **FinBERT model** to analyze news headlines and generate **Buy / Hold / Sell sentiment signals** for selected stocks.

The application collects financial news from multiple sources, processes the headlines using **FinBERT (a BERT model trained for financial text)**, and displays the sentiment results in a **Streamlit dashboard**.

The model files are **not included in this repository** because they exceed GitHub’s file size limits. Instead, they are automatically generated/downloaded when running the setup script.

---

# Project Structure

```
sentiment_project1
│
├── app2.py              # Streamlit dashboard
├── finbert.py           # Script that downloads/creates the FinBERT model
├── all-data.csv         # Dataset used by the model
├── .gitignore           # Prevents large or unnecessary files from uploading
│
├── finbert_model/       # Generated automatically (not included in repo)
├── venv/                # Virtual environment (ignored)
└── __pycache__/         # Python cache (ignored)
```

Ignored files and folders (not pushed to GitHub):

```
venv/
__pycache__/
finbert_model/
.env
*.pyc
```

---

# Features

* Financial news aggregation
* Sentiment analysis using FinBERT
* Stock sentiment scoring
* Buy / Hold / Sell recommendations
* Interactive Streamlit dashboard
* Real-time news fetching from APIs

---

# Requirements

Install Python **3.9+** before running the project.

Required Python packages:

```
streamlit
transformers
torch
yfinance
feedparser
streamlit-autorefresh
requests
numpy
```

---

# Setup Instructions

Follow these steps to run the project locally.

---

## Step 1 — Clone the repository

```
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

```
cd YOUR_REPOSITORY
```

---

## Step 2 — Create a virtual environment

```
python -m venv venv
```

Activate the environment.

### Windows

```
venv\Scripts\activate
```

### Mac / Linux

```
source venv/bin/activate
```

---

## Step 3 — Install dependencies

```
pip install streamlit transformers torch yfinance feedparser streamlit-autorefresh requests numpy
```

---

## Step 4 — Generate the FinBERT model

The repository does **not contain the FinBERT model files**.

Run the following script to automatically download and create them:

```
python finbert.py
```

This will create:

```
finbert_model/
```

containing:

```
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.txt
```

---

## Step 5 — Run the dashboard

Start the Streamlit app:

```
streamlit run app2.py
```

---

## Step 6 — Open the dashboard

After running the command, Streamlit will display a local URL such as:

```
http://localhost:8501
```

Open it in your browser to view the dashboard.

---

# How the System Works

1. News headlines are collected from multiple sources.
2. Headlines are cleaned and filtered.
3. FinBERT processes each headline.
4. Sentiment scores are calculated.
5. Average sentiment determines:

```
Positive sentiment → Buy
Negative sentiment → Sell
Neutral sentiment → Hold
```

Results are displayed in the dashboard.

---

# Notes

* Large files such as model weights are excluded from GitHub using `.gitignore`.
* The FinBERT model is generated locally when running `finbert.py`.
* API keys may be required depending on the news sources used.

---

# Future Improvements

Possible enhancements:

* Sector-wise sentiment heatmap
* Sentiment trend analysis
* Market-wide sentiment tracking
* GPT-based financial news summarization
* Deployment using Streamlit Cloud or Docker

---

# License

This project is intended for **educational and research purposes**.
