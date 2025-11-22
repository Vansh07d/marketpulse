import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests
import numpy as np
from datetime import datetime
import feedparser
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

MODEL_PATH = "finbert_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

API_KEY_NEWSDATA = 'pub_e2e341280d184b689583d52fa1358868'

synonyms_map = {
    'INFY': ['Infosys', 'Infosys Technologies'],
    'MARUTI': ['Maruti Suzuki', 'Maruti Suzuki India', 'Maruti'],
    'TATAMOTORS': ['Tata Motors Ltd', 'Tata Motors'],
    'RELIANCE': ['Reliance Industries', 'Reliance'],
    'HDFCBANK': ['HDFC Bank'],
    'ICICIBANK': ['ICICI Bank']
}

def requests_session_with_retries(total_retries=3, backoff_factor=0.3):
    session = requests.Session()
    retries = Retry(total=total_retries,
                    backoff_factor=backoff_factor,
                    status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

session = requests_session_with_retries()

@st.cache_data(ttl=600)
def get_moneycontrol_rss_news():
    url = 'https://www.moneycontrol.com/rss/markets.xml'
    try:
        feed = feedparser.parse(url)
        return [entry.title for entry in feed.entries[:5]]
    except:
        return []

@st.cache_data(ttl=600)
def get_google_news_rss(company_name):
    search = company_name.replace(' ', '+') + '+NSE'
    url = f"https://news.google.com/rss/search?q={search}"
    try:
        feed = feedparser.parse(url)
        return [entry.title for entry in feed.entries[:5]]
    except:
        return []

@st.cache_data(ttl=600)
def get_newsdata_news(company_name, api_key):
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={company_name}&language=en"
    try:
        resp = session.get(url, timeout=10)
        data = resp.json()
        results = data.get('results', [])
        if results:
            return [item.get('title', '') for item in results[:5]]
        return []
    except:
        return []

@st.cache_data(ttl=600)
def finbert_sentiment(texts):
    sentiment_labels = ['negative', 'neutral', 'positive']
    scores = []
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        score = (-1 * probs[0].item()) + (1 * probs[2].item())
        scores.append(score)
        pred = sentiment_labels[torch.argmax(probs).item()]
        results.append({'text': text, 'score': score, 'pred': pred})
    avg_score = float(np.mean(scores)) if scores else 0.0
    return avg_score, results

def buy_decision(avg_score, threshold=0.15):
    if avg_score > threshold:
        return ("Buy", "🟢")
    elif avg_score < -threshold:
        return ("Sell", "🔴")
    else:
        return ("Hold", "🟡")

tickers = {
    'AAPL': 'AAPL',
    'MSFT': 'MSFT',
    'NVDA': 'NVDA',
    'GOOGL': 'GOOGL',
    'AMZN': 'AMZN',
    'META': 'META',
    'TSLA': 'TSLA',
    'NFLX': 'NFLX',
    'JPM': 'JPM',
    'BRK.B': 'BRK-B',
    'TCS': 'TCS.NS',
    'RELIANCE': 'RELIANCE.NS',
    'SBI': 'SBIN.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'INFY': 'INFY.NS',
    'MARUTI': 'MARUTI.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'TATAMOTORS': 'TATAMOTORS.NS',
    'AXISBANK': 'AXISBANK.NS',
    'BHEL': 'BHEL.NS',
    'ADANIPOWER': 'ADANIPOWER.NS'
}

st.set_page_config(page_title="Financial Sentiment Dashboard", layout="wide")
st.sidebar.title("🔍 Search Stocks")
search_term = st.sidebar.text_input("Enter stock name or ticker", "").strip().lower()

if search_term:
    filtered_tickers = {name: symbol for name, symbol in tickers.items() if search_term in name.lower()}
else:
    filtered_tickers = tickers

st.sidebar.title("🔄 Refresh")
if st.sidebar.button("Refresh Dashboard"):
    st.rerun()

st.title("📊 Real-Time Financial Sentiment Dashboard (US & India)")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

stock_items = list(filtered_tickers.items())
cols_per_row = 4

for i in range(0, len(stock_items), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, (name, symbol) in enumerate(stock_items[i:i+cols_per_row]):
        with cols[j]:
            with st.spinner(f"Loading news for {name}..."):
                headlines = []
                headlines += get_moneycontrol_rss_news()
                synonyms = synonyms_map.get(name, [name])
                for query in synonyms:
                    headlines += get_google_news_rss(query)
                    headlines += get_newsdata_news(query, API_KEY_NEWSDATA)
                unique_headlines = list(set(headlines))
                avg_score, sentiment_results = finbert_sentiment(unique_headlines)

            decision, color = buy_decision(avg_score)

            st.markdown(f"### {name} ({symbol})")
            st.markdown(f"**Recommendation**: `{decision} {color}`")

            st.markdown(
                f"**Avg Sentiment**: "
                f"<span style='color:{'green' if avg_score>0.15 else 'red' if avg_score<-0.15 else 'gray'};'>{avg_score:.3f}</span>",
                unsafe_allow_html=True
            )

            st.progress(int((avg_score + 1) / 2 * 100))

            with st.expander("📰 Latest Headlines & Sentiment"):
                if sentiment_results:
                    for res in sentiment_results[:5]:
                        st.write(f"• {res['text']} ({res['pred']}, {res['score']:.2f})")
                else:
                    st.write("No news available.")
            st.write("---")
