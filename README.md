# stock-price-forecasting
This project combines Hidden Markov Models (HMMs) and Bidirectional LSTMs (Long Short-Term Memory Networks) to forecast stock prices using both statistical regime modeling and deep learning sequence prediction.

It also includes a fully functional Tkinter GUI that lets you backtest forecasts visually — showing predicted prices, actual prices, and market regimes identified by the HMM model.
The model combines a Gaussian Hidden Markov Model (HMM) for regime detection with a Bidirectional LSTM–GRU network for short-term price prediction. The HMM model uses financial indicators such as RSI, MACD, Volatility, Moving Averages & Golden Cross indicator and those can be chosen manually under the variable "obs".
