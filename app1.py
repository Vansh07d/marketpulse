import streamlit as st
from engine import predict_sentiment, analyze_chunk_mode
from realtime import fetch_tweets, fetch_news 

st.set_page_config(page_title="FinBERT Sentiment Analysis", layout="wide")
st.title("📊 FinBERT Financial Sentiment Analyzer")

source = st.radio("Choose data source:", ["Manual Input", "Twitter", "News"])

if source == "Manual Input":
    mode = st.radio("Choose analysis mode:", ["Overall paragraph", "Sentence-by-sentence (chunk)"])
    user_text = st.text_area("Enter financial text to analyze:", height=200)
    uploaded_file = st.file_uploader("Or upload a .txt file for batch analysis:", type=["txt"])

    if uploaded_file is not None:
        try:
            file_text = uploaded_file.read().decode("utf-8")
            user_text = file_text.strip()
            st.success("File loaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if st.button("Analyze Sentiment"):
        if not user_text.strip():
            st.warning("Please enter some text or upload a .txt file.")
        else:
            if mode == "Overall paragraph":
                res = predict_sentiment(user_text)
                st.subheader("Overall Paragraph Sentiment")
                st.write({k: f"{v*100:.2f}%" for k, v in res.items()})
                st.success(f"Predicted Label: {max(res, key=res.get)}")
            else:
                sentences, all_results, avg_probs = analyze_chunk_mode(user_text)
                st.subheader("Sentence-by-Sentence Analysis")
                for sent, res in zip(sentences, all_results):
                    st.write(f"**{sent}**")
                    st.write({k: f"{v*100:.2f}%" for k, v in res.items()})
                    st.markdown("---")
                st.subheader("Average Sentiment Across Sentences")
                st.write({k: f"{v*100:.2f}%" for k, v in avg_probs.items()})
                st.success(f"Overall Predicted Label: {max(avg_probs, key=avg_probs.get)}")

elif source == "Twitter":
    query = st.text_input("Enter keyword or hashtag (e.g. #stocks or Tesla):")
    count = st.slider("Number of Tweets", 5, 50, 10)

    if st.button("Fetch & Analyze Tweets"):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            tweets = fetch_tweets(query, count)
            if not tweets:
                st.warning("No tweets fetched for this query.")
            else:
                st.subheader(f"Fetched {len(tweets)} Tweets")
                avg_scores = {"positive": 0, "neutral": 0, "negative": 0}

                for tw in tweets:
                    res = predict_sentiment(tw)
                    for k in avg_scores:
                        avg_scores[k] += res.get(k, 0)
                    st.write(f"**{tw}**")
                    st.write({k: f"{v*100:.2f}%" for k, v in res.items()})
                    st.markdown("---")

                avg_scores = {k: v / len(tweets) for k, v in avg_scores.items()}
                st.subheader("Average Sentiment Across Tweets")
                st.write({k: f"{v*100:.2f}%" for k, v in avg_scores.items()})
                st.success(f"Overall Predicted Label: {max(avg_scores, key=avg_scores.get)}")

elif source == "News":
    query = st.text_input("Enter news topic (e.g. stocks, finance, bitcoin):")
    count = st.slider("Number of Articles", 5, 30, 10)

    if st.button("Fetch & Analyze News"):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            articles = fetch_news(query, count)
            if not articles:
                st.warning("No news articles fetched for this query.")
            else:
                st.subheader(f"Fetched {len(articles)} Articles")
                avg_scores = {"positive": 0, "neutral": 0, "negative": 0}

                for art in articles:
                    res = predict_sentiment(art)
                    for k in avg_scores:
                        avg_scores[k] += res.get(k, 0)
                    st.write(f"**{art}**")
                    st.write({k: f"{v*100:.2f}%" for k, v in res.items()})
                    st.markdown("---")

                avg_scores = {k: v / len(articles) for k, v in avg_scores.items()}
                st.subheader("Average Sentiment Across Articles")
                st.write({k: f"{v*100:.2f}%" for k, v in avg_scores.items()})
                st.success(f"Overall Predicted Label: {max(avg_scores, key=avg_scores.get)}")
