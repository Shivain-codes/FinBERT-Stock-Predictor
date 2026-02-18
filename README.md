


## 🚀 Features
- **FinBERT Sentiment Analysis**: Pre-trained BERT for financial text from news APIs (e.g., yfinance, yahooquery). [web:1][web:15]
- **Hybrid Prediction**: Combines sentiment embeddings with time-series data using LSTM or regression. [web:19][web:24]
- **Data Pipeline**: Fetch stock data, clean news, compute MST/PMFG graphs for visualization. [web:7]
- **Evaluation**: MAE, MAPE, accuracy metrics with backtesting on NASDAQ-100. [web:21]
- **Deployment-Ready**: Google Colab notebooks, Streamlit app for interactive predictions. [web:7]

## 📊 Quick Example
Predict next-day movement for AAPL:

python
import yfinance as yf
from transformers import pipeline
from finbert_model import predict_stock  # Your custom module

# Fetch data
data = yf.download('AAPL', period='1mo')
news = fetch_news('AAPL')  # Custom news fetcher

# Sentiment + Prediction
sentiment = pipeline('sentiment-analysis', model='ProsusAI/finbert')(news)

torch transformers
pandas numpy yfinance yahooquery
scikit-learn lightgbm networkx matplotlib

├── notebooks/
│   ├── 01_data_fetching.ipynb      # yfinance + news
│   ├── 02_sentiment_analysis.ipynb # FinBERT processing
│   └── 03_prediction_lstm.ipynb    # Model training
├── src/
│   ├── finbert_sentiment.py
│   └── stock_predictor.py
├── data/                           # Sample datasets
├── models/                         # Trained weights
└── app.py                          # Streamlit demo


prediction = predict_stock(data, sentiment)
print(f"Predicted change: {prediction:.2f}%")
