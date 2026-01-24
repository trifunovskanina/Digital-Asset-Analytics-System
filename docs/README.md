# Digital Asset Analytics System



## Market Data Foundation

The system is built on Open, High, Low, Close, Volume market data for up
to ten years of historical activity of approximately 1000 cryptocurrencies.

* Opening Price
* High and Low 
* Closing Price
* Trading Volume


## Long Short-Term Memory Closing Price Forecast

A Long Short-Term Memory (LSTM) neural network is trained as a univariate time-series regression problem, where sequences of past closing prices are used to predict the next time step.

A sliding window (lookback) approach is applied, and the network consists of stacked LSTM layers followed by a fully connected output layer.

* Uses historical closing prices only
* Preserves temporal order in training and evaluation
* Supports closing price forecasting
* Demonstrates sequence learning with deep neural networks


## Natural Language Processing Transformer Sentiment Analysis

News sentiment analysis is performed using a pretrained transformer natural language processing model.

Recent cryptocurrency news articles are retrieved from the database and analyzed using a DistilBERT model fine-tuned on the Stanford Sentiment Treebank (SST-2).

Each article is classified as positive or negative, along with a confidence score.

* Uses a pretrained transformer model (DistilBERT)
* Performs sentiment classification
* Produces sentiment labels and scores for each article


## Technical Analysis

Focuses on identifying patterns based on:

Oscillators

* RSI (Relative Strength Index)
* MACD (Moving Average Convergence Divergence)
* Stochastic Oscillator
* ADX (Average Directional Index)
* CCI (Commodity Channel Index))

Moving Averages

* SMA (Simple Moving Average)
* EMA (Exponential Moving Average)
* WMA (Weighted Moving Average)
* Bollinger Bands
* Volume Moving Averages