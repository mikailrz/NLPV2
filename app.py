import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
import torch
import nest_asyncio
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Enable nested event loops for Kaggle/Jupyter
nest_asyncio.apply()

st.title("📰 NLP-Based Business News Analyzer")
st.write("This app fetches live business news, analyzes sentiment, and generates key insights.")

# ---- News Sources ----
NEWS_SOURCES = [
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"},
    {"url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"},
    {"url": "https://www.scmp.com/rss/91/feed"},
    {"url": "https://www.theguardian.com/uk/business/rss"},
    {"url": "https://www.nasdaq.com/feed/rssoutbound?category=Markets"},
]

# ---- Asynchronous Fetch News ----
async def fetch_url(session, url):
    try:
        async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10) as response:
            if response.status == 200:
                content = await response.text()
                soup = BeautifulSoup(content, "lxml-xml")
                articles = []
                for item in soup.find_all("item"):
                    title = item.title.text if item.title else "No Title"
                    description = item.description.text if item.description else "No Description"
                    link = item.link.text if item.link else None
                    articles.append({"title": title, "description": description, "link": link})
                return articles
    except Exception as e:
        return []

async def fetch_news():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, source["url"]) for source in NEWS_SOURCES]
        results = await asyncio.gather(*tasks)
        all_articles = [article for sublist in results for article in sublist]
    return pd.DataFrame(all_articles)

def get_news():
    return asyncio.run(fetch_news())

st.write("📡 Fetching Latest News...")
news_df = get_news()

if news_df.empty:
    st.warning("No news articles fetched.")
else:
    st.success(f"✅ {len(news_df)} news articles fetched!")

# ---- Sentiment Analysis ----
def analyze_sentiment(news_df):
    device = 0 if torch.cuda.is_available() else -1
    sentiment_model = pipeline("text-classification", model="ProsusAI/finbert", device=device)
    news_df[['sentiment', 'sentiment_score']] = news_df['description'].apply(
        lambda x: pd.Series((sentiment_model(x[:512])[0]['label'], sentiment_model(x[:512])[0]['score']))
    )
    return news_df

news_df = analyze_sentiment(news_df)

# ---- Word Cloud with Enhanced Cleaning ----
st.subheader("☁️ Key Business Terms in News Descriptions")
DEFAULT_STOPWORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = {"description", "news", "report", "article", "said", "also", "will", "new"}
ALL_STOPWORDS = DEFAULT_STOPWORDS.union(CUSTOM_STOPWORDS)

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.lower().split()
    words = [word for word in words if word not in ALL_STOPWORDS and len(word) > 3]
    return ' '.join(words)

cleaned_text = " ".join(news_df['description'].dropna().apply(clean_text))

wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color='white',
    colormap='Blues',
    stopwords=ALL_STOPWORDS
).generate(cleaned_text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# ---- Sentiment Visualization ----
st.subheader("📊 Sentiment Distribution")
sentiment_counts = news_df['sentiment'].value_counts()
fig = px.pie(
    sentiment_counts,
    names=sentiment_counts.index,
    values=sentiment_counts.values,
    title="Sentiment Distribution (Positive/Neutral/Negative)",
    hole=0.3,
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig)

# ---- Display Focused News Articles ----
st.subheader("🗞️ Latest News Articles")
KEY_TERMS = ['stock', 'market', 'economy', 'investment', 'bank', 'trade', 'currency', 'tech']
filtered_df = news_df[news_df['description'].str.contains('|'.join(KEY_TERMS), case=False, na=False)]

for _, row in filtered_df.iterrows():
    st.markdown(f"### {row['title']}")
    st.write(f"**Description**: {row['description'][:200]}...")  # Truncate for readability
    st.write(f"**Sentiment**: {row['sentiment']} (Score: {row['sentiment_score']:.2f})")
    st.markdown(f"[Read More]({row['link']})")
    st.write("---")

st.success("🚀 Analysis Complete: Focused on Financial/Business News!")
