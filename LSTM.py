import os

os.environ["OMP_NUM_THREADS"] = "2"

import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hmmlearn.hmm import GaussianHMM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.model_selection import train_test_split
import threading
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Environment & Seeding
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


#  Technical Indicators
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    return macd.fillna(0)


def compute_volatility(series, window=10):
    return series.pct_change().rolling(window).std().fillna(0)


#  LSTM Sequence Creation
def create_sequences(data, seq_len=25):
    X, y_price, y_dir = [], [], []
    for i in range(len(data) - seq_len):
        seq = data[i:i + seq_len]
        X.append(seq)
        y_price.append(data[i + seq_len, 0])
        direction = 1 if data[i + seq_len, 0] > data[i + seq_len - 1, 0] else 0
        y_dir.append(direction)
    return np.array(X), np.array(y_price), np.array(y_dir)


# Inverse scale prices
def inverse_scale_prices(scaled_prices, scaler, feature_idx=0):
    dummy = np.zeros((len(scaled_prices), scaler.scale_.shape[0]))
    dummy[:, feature_idx] = scaled_prices.flatten()
    inv = scaler.inverse_transform(dummy)[:, feature_idx]
    return inv


def log_progress(message):
    progress_text.insert(tk.END, message + "\n")
    progress_text.see(tk.END)
    progress_text.update()


# === Main prediction function===
def run_prediction_with_logging(ticker, start_date, end_date, seq_len=60, future_steps=30):
    log_progress(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    df = df[['Close', 'Volume']].dropna()
    df.columns = df.columns.get_level_values(0)
    log_progress("Computing technical indicators...")
    df['MA_short'] = df['Close'].rolling(window=10).mean()
    df['MA_long'] = df['Close'].rolling(window=40).mean()
    df['golden_cross'] = (df['MA_short'] > df['MA_long']).astype(int)
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df['Volatility'] = compute_volatility(df['Close'])
    df.dropna(inplace=True)



    log_progress("Training HMM...")
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    df = df.loc[log_returns.index]
    obs = np.column_stack([
        log_returns,
        df['MACD'],
        df['Volatility']
    ])
    scaler_hmm = StandardScaler()
    obs_scaled = scaler_hmm.fit_transform(obs)
    hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=500, random_state=SEED)
    hmm.fit(obs_scaled)
    states = hmm.predict(obs_scaled)
    df['State'] = states
    state_means = pd.DataFrame({
        'State': np.arange(hmm.n_components),
        'MeanReturn': [log_returns[states == i].mean() for i in range(hmm.n_components)]
    })
    state_order = state_means.sort_values('MeanReturn').reset_index(drop=True)
    state_mapping = {int(old): int(new) for new, old in enumerate(state_order['State'])}
    df['State'] = df['State'].map(state_mapping)


    log_progress("Preparing LSTM data...")
    features = ['Close', 'State']
    scaler_lstm = MinMaxScaler()
    data_scaled = scaler_lstm.fit_transform(df[features].values)
    X, y_price, y_dir = create_sequences(data_scaled, seq_len)

    log_progress("Splitting data for training...")
    X_train, X_val, y_price_train, y_price_val, y_dir_train, y_dir_val = train_test_split(
        X, y_price, y_dir, test_size=0.15, shuffle=False)


    log_progress("Building LSTM model...")
    inputs = Input(shape=(seq_len, X.shape[2]))
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = GRU(32, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    out_price = Dense(1, name='price_output')(x)
    out_dir = Dense(1, activation='sigmoid', name='direction_output')(x)
    model = Model(inputs, [out_price, out_dir])
    model.compile(optimizer=Adam(1e-3),
                  loss={'price_output': 'mse', 'direction_output': 'binary_crossentropy'},
                  loss_weights={'price_output': 0.8, 'direction_output': 0.2},
                  metrics={'direction_output': 'accuracy'})

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=0)

    log_progress("Training LSTM model...")
    model.fit(X_train,
              {'price_output': y_price_train, 'direction_output': y_dir_train},
              validation_data=(X_val, {'price_output': y_price_val, 'direction_output': y_dir_val}),
              epochs=100,
              batch_size=32,
              callbacks=[early_stop, reduce_lr],
              verbose=1)

    log_progress("Making predictions...")
    pred_price_val_scaled, _ = model.predict(X_val)
    pred_price_val = inverse_scale_prices(pred_price_val_scaled, scaler_lstm)
    actual_price_val = inverse_scale_prices(y_price_val, scaler_lstm)

    log_progress("Forecasting future prices...")
    last_seq = X[-1]
    future_prices_scaled = []
    for i in range(future_steps):
        inp = last_seq.reshape(1, seq_len, X.shape[2])
        p, _ = model.predict(inp)
        future_prices_scaled.append(p[0][0])
        log_progress(f"Day {i + 1} forecast: {inverse_scale_prices(np.array([p[0][0]]), scaler_lstm)[0]:.2f}")
        new_point = np.array([list(last_seq[-1])])
        new_point[0][0] = p[0][0]
        last_seq = np.vstack([last_seq[1:], new_point])

    future_prices = inverse_scale_prices(np.array(future_prices_scaled), scaler_lstm)
    val_dates = df.index[-len(y_price_val):]
    future_dates = pd.date_range(start=val_dates[-1] + pd.Timedelta(days=1),
                                 periods=future_steps, freq='B')

    log_progress("Prediction complete!")
    return val_dates, actual_price_val, pred_price_val, future_dates, future_prices, df['State']


def threaded_prediction():
    threading.Thread(target=run_and_plot).start()


def run_and_plot():
    val_dates, actual_val, pred_val, future_dates, future_prices, states = run_prediction_with_logging(
        ticker=ticker_entry.get().upper(),
        start_date=start_entry.get(),
        end_date=date.today(),
        future_steps=int(prediction_days.get())
    )


    for widget in plot_frame.winfo_children():
        widget.destroy()


    def aggregate_states_mode(states, chunk_size=30):
        states_arr = np.array(states)
        n_chunks = len(states_arr) // chunk_size
        agg_states = []

        for i in range(n_chunks):
            chunk = states_arr[i*chunk_size : (i+1)*chunk_size]
            mode_state = pd.Series(chunk).mode()[0]
            agg_states.append(mode_state)

        remainder = len(states_arr) % chunk_size
        if remainder > 0:
            chunk = states_arr[-remainder:]
            mode_state = pd.Series(chunk).mode()[0]
            agg_states.append(mode_state)

        return np.array(agg_states)

    chunk_size = 5
    states_val = states[-len(val_dates):]  # align with validation dates
    agg_states = aggregate_states_mode(states_val, chunk_size)


    agg_dates = [val_dates[i*chunk_size] for i in range(len(agg_states))]
 

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(val_dates, actual_val, label='Actual Price', color='blue')
    ax.plot(val_dates, pred_val, label='Predicted Price', color='orange')
    ax.plot(future_dates, future_prices, linestyle='--', label='Forecasted Price', color='green')

    # --- State colors ---
    colors = ['#ff6666', '#ffeb66', '#66ff66']  # Red, Yellow, Green
    alpha_value = 0.3


    for i, state in enumerate(agg_states):
        start_date = agg_dates[i]

        if i < len(agg_states) - 1:
            end_date = agg_dates[i+1] - pd.Timedelta(days=1)
        else:
            end_date = val_dates[-1]
        ax.axvspan(start_date, end_date, color=colors[state], alpha=alpha_value)


    last_state = agg_states[-1]
    ax.axvspan(future_dates[0], future_dates[-1], color=colors[last_state], alpha=0.15)


    ax.set_title(f"{ticker_entry.get().upper()} Price Prediction & Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)


    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)


# --- GUI Layout ---
root = tk.Tk()
root.title("Stock Prediction GUI with Progress")
root.geometry("1000x700")

input_frame = ttk.Frame(root)
input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

ttk.Label(input_frame, text="Ticker:").pack(side=tk.LEFT)
ticker_entry = ttk.Entry(input_frame)
ticker_entry.insert(0, "META")
ticker_entry.pack(side=tk.LEFT, padx=5)

ttk.Label(input_frame, text="Start Date (YYYY-MM-DD):").pack(side=tk.LEFT, padx=5)
start_entry = ttk.Entry(input_frame)
start_entry.insert(0, "2018-01-01")
start_entry.pack(side=tk.LEFT)

ttk.Label(input_frame, text="days of prediction").pack(side=tk.LEFT, padx=5)
prediction_days = ttk.Entry(input_frame)
prediction_days.insert(0, str(30))
prediction_days.pack(side=tk.LEFT, padx=5)

ttk.Button(input_frame, text="Run Prediction", command=threaded_prediction).pack(side=tk.LEFT, padx=10)

plot_frame = ttk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=1, padx=10, pady=10)

progress_text = tk.Text(root, height=7)
progress_text.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

root.mainloop()
