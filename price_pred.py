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

# === Environment & Seeding ===
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# === Technical Indicators ===
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


# === LSTM Sequence Creation ===
def create_sequences(data, seq_len=25):
    X, y_price, y_dir = [], [], []
    for i in range(len(data) - seq_len):
        seq = data[i:i + seq_len]
        X.append(seq)
        # target is the next-step RETURN (feature index 0)
        y_price.append(data[i + seq_len, 0])
        direction = 1 if data[i + seq_len, 0] > data[i + seq_len - 1, 0] else 0
        y_dir.append(direction)
    return np.array(X), np.array(y_price), np.array(y_dir)


# === Inverse scale helper (works for scalers fit on multiple features) ===
def inverse_scale_feature(scaled_array, scaler, feature_idx=0):
    """Inverse transform a single feature column that was scaled together with other features."""
    arr = np.array(scaled_array).reshape(-1)
    dummy = np.zeros((len(arr), scaler.scale_.shape[0]))
    dummy[:, feature_idx] = arr.flatten()
    inv = scaler.inverse_transform(dummy)[:, feature_idx]
    return inv


# === Progress logging ===
def log_progress(message):
    try:
        progress_text.insert(tk.END, message + "\n")
        progress_text.see(tk.END)
        progress_text.update()
    except Exception:
        # If GUI not ready, fallback to print
        print(message)


# === Main prediction function (forecast into past) ===
def run_prediction_with_logging(ticker, start_date, end_date, seq_len=60, forecast_days=30):
    log_progress(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if df.empty:
        raise ValueError("No data downloaded for ticker. Check ticker or date range.")
    df = df[['Close', 'Volume']].dropna()
    # ensure simple columns if multiindex
    df.columns = df.columns.get_level_values(0)

    # Compute returns (stationary target) and technical indicators
    df['Return'] = df['Close'].pct_change().fillna(0)
    log_progress("Computing technical indicators...")
    df['MA_short'] = df['Close'].rolling(window=10).mean()
    df['MA_long'] = df['Close'].rolling(window=40).mean()
    df['golden_cross'] = (df['MA_short'] > df['MA_long']).astype(int)
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df['Volatility'] = compute_volatility(df['Close'])
    df.dropna(inplace=True)

    # --- HMM (keeps using log returns + indicators) ---
    log_progress("Training HMM...")
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    df = df.loc[log_returns.index]  # align
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

    # Order states by mean return and remap to 0..n-1 by ascending mean
    state_means = pd.DataFrame({
        'State': np.arange(hmm.n_components),
        'MeanReturn': [log_returns[states == i].mean() for i in range(hmm.n_components)]
    })
    state_order = state_means.sort_values('MeanReturn').reset_index(drop=True)
    state_mapping = {int(old): int(new) for new, old in enumerate(state_order['State'])}
    df['State'] = df['State'].map(state_mapping)

    # --- Split for forecast backtest ---
    if forecast_days <= 0:
        raise ValueError("forecast_days must be > 0")
    if forecast_days + seq_len >= len(df):
        raise ValueError("Not enough data for the requested forecast_days + seq_len. Reduce forecast_days or use earlier start date.")

    log_progress("Splitting data (simulate forecast on past segment)...")
    train_df = df.iloc[:-forecast_days]
    test_df = df.iloc[-(forecast_days + seq_len):]

    # Use Return and State as features; feature 0 (Return) is the target
    features = ['Return', 'State']
    # use range (-1,1) for returns to allow negative values properly
    scaler_lstm = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler_lstm.fit_transform(train_df[features].values)
    X, y_price, y_dir = create_sequences(data_scaled, seq_len)

    if len(X) == 0:
        raise ValueError("Not enough training data to create sequences. Increase data window or reduce seq_len.")

    X_train, X_val, y_price_train, y_price_val, y_dir_train, y_dir_val = train_test_split(
        X, y_price, y_dir, test_size=0.15, shuffle=False)

    # --- Build LSTM model ---
    log_progress("Building LSTM model...")
    inputs = Input(shape=(seq_len, X.shape[2]))
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = GRU(32, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    out_price = Dense(1, name='price_output')(x)            # predicts scaled RETURN (feature 0)
    out_dir = Dense(1, activation='sigmoid', name='direction_output')(x)
    model = Model(inputs, [out_price, out_dir])
    model.compile(optimizer=Adam(1e-3),
                  loss={'price_output': 'mse', 'direction_output': 'binary_crossentropy'},
                  loss_weights={'price_output': 0.8, 'direction_output': 0.2},
                  metrics={'direction_output': 'accuracy'})

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)

    log_progress("Training LSTM model...")
    model.fit(X_train,
              {'price_output': y_price_train, 'direction_output': y_dir_train},
              validation_data=(X_val, {'price_output': y_price_val, 'direction_output': y_dir_val}),
              epochs=120,
              batch_size=32,
              callbacks=[early_stop, reduce_lr],
              verbose=1)

    # --- Backtest forecast into the past ---
    log_progress("Simulating past forecast...")
    test_scaled = scaler_lstm.transform(test_df[features].values)
    last_seq = test_scaled[:seq_len].copy()  # initial sequence for rolling forecast
    # actual_test_prices are the actual closes corresponding to predicted days
    actual_test_prices = test_df['Close'].values[seq_len:]
    forecast_scaled = []

    for i in range(forecast_days):
        inp = last_seq.reshape(1, seq_len, test_scaled.shape[1])
        p, _ = model.predict(inp, verbose=1)
        forecast_scaled.append(p[0][0])  # this is the SCALED return (feature 0 scaled)
        # roll the sequence with the predicted scaled return (put into feature 0) and keep state same as last row
        new_point = last_seq[-1].copy()
        new_point[0] = p[0][0]
        last_seq = np.vstack([last_seq[1:], new_point])
        log_progress(f"Predicted day {i + 1}/{forecast_days}")

    # inverse-scale predicted returns back to raw returns
    forecast_returns = inverse_scale_feature(np.array(forecast_scaled), scaler_lstm, feature_idx=0)

    # reconstruct forecasted prices from returns
    # starting price = the close immediately before the first predicted day (that's test_df Close at index seq_len-1)
    start_price = test_df['Close'].values[seq_len - 1]
    reconstructed_prices = []
    price = start_price
    for r in forecast_returns:
        price = price * (1 + r)
        reconstructed_prices.append(price)
    reconstructed_prices = np.array(reconstructed_prices)

    forecast_dates = test_df.index[seq_len:]  # dates that correspond to predictions

    log_progress("Backtest forecast complete!")
    # return forecast dates, actual prices for those dates, forecasted prices (reconstructed), states series, and full df
    return forecast_dates, actual_test_prices, reconstructed_prices, df['State'], df


# --- Thread wrapper ---
def threaded_prediction():
    threading.Thread(target=run_and_plot).start()


def run_and_plot():
    ticker = ticker_entry.get().upper()
    start_date = start_entry.get()

    # Run model and get predictions + df + states
    val_dates, actual_val, pred_val, states, df = run_prediction_with_logging(
        ticker=ticker,
        start_date=start_date,
        end_date=date.today(),
        seq_len=int(seq_len_entry.get()),
        forecast_days=int(prediction_days.get())
    )

    # --- Prepare 1-year window ---
    df.index = pd.to_datetime(df.index)
    one_year_cutoff = df.index.max() - pd.Timedelta(days=365)
    one_year_data = df[df.index >= one_year_cutoff]
    one_year_states = states[-len(one_year_data):]  # align states with 1-year window

    # --- Compute mean return for each state ---
    returns = df['Close'].pct_change().fillna(0).values
    state_means = {}
    for s in np.unique(states):
        state_means[s] = np.mean(returns[np.array(states) == s])

    # Sort states by average return
    sorted_states = sorted(state_means.items(), key=lambda x: x[1])
    state_rank = {s: rank for rank, (s, _) in enumerate(sorted_states)}

    # Assign color dynamically
    color_map = {
        0: (1, 0, 0, 0.25),  # lowest return → red
        1: (1, 1, 0, 0.25),  # middle → yellow
        2: (0, 1, 0, 0.25)   # highest → green
    }

    # Clear previous plot
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Debug prints (helpful)
    print("one_year date range:", one_year_data.index.min(), one_year_data.index.max())
    print("val_dates range:", val_dates.min(), val_dates.max())
    print("len(val_dates), len(pred_val), len(actual_val):", len(val_dates), len(pred_val), len(actual_val))
    print("pred_val sample (first 5):", pred_val[:5])
    print("forecast_prices min/max:", np.min(pred_val), np.max(pred_val))
    print("forecast_prices vs actual (first 5):", np.round(pred_val[:5], 2), np.round(actual_val[:5], 2))

    # Clip predictions to plotting window (in case)
    val_dates = pd.to_datetime(val_dates)
    mask_in_one_year = (val_dates >= one_year_data.index.min()) & (val_dates <= one_year_data.index.max())
    if not mask_in_one_year.any():
        print("Warning: predicted dates are outside the 1-year plotting window; plotting anyway.")
        val_dates_plot = val_dates
        pred_val_plot = pred_val
        actual_val_plot = actual_val
    else:
        val_dates_plot = val_dates[mask_in_one_year]
        pred_val_plot = np.array(pred_val)[mask_in_one_year]
        actual_val_plot = np.array(actual_val)[mask_in_one_year]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Shade states dynamically (send behind lines with zorder=0) ---
    unique_states = np.unique(one_year_states)
    for s in unique_states:
        rank = state_rank[s]
        mask = one_year_states == s
        ax.fill_between(
            one_year_data.index,
            one_year_data['Close'].min(),
            one_year_data['Close'].max(),
            where=mask,
            color=color_map.get(rank, (0.5, 0.5, 0.5, 0.12)),
            alpha=0.25,
            zorder=0
        )

    # --- Plot prices (put lines on top with higher zorder) ---
    ax.plot(one_year_data.index, one_year_data['Close'], label='Actual Price', color='blue', linewidth=2, zorder=3)
    # show actuals in forecast window with dashed black
    ax.plot(val_dates_plot, actual_val_plot, label='Actual (forecast window)', color='black', linewidth=1.0, linestyle='--', zorder=3)
    # predicted line with markers to make sparse segments visible
    ax.plot(val_dates_plot, pred_val_plot, label='Predicted', color='orange', linewidth=2.5, marker='o', markersize=3, zorder=4)

    # --- Styling ---
    ax.set_title(f"{ticker_entry.get().upper()} — Forecast & HMM States (Auto Classified, Last 1 Year)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # Embed in Tkinter GUI
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)


# --- GUI Layout ---
root = tk.Tk()
root.title("Stock Forecast Backtest GUI (returns-based)")
root.geometry("1200x800")

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

ttk.Label(input_frame, text="Seq length:").pack(side=tk.LEFT, padx=5)
seq_len_entry = ttk.Entry(input_frame, width=6)
seq_len_entry.insert(0, "60")
seq_len_entry.pack(side=tk.LEFT)

ttk.Label(input_frame, text="Days to backtest:").pack(side=tk.LEFT, padx=5)
prediction_days = ttk.Entry(input_frame)
prediction_days.insert(0, str(60))
prediction_days.pack(side=tk.LEFT, padx=5)

ttk.Button(input_frame, text="Run Backtest", command=threaded_prediction).pack(side=tk.LEFT, padx=10)

plot_frame = ttk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=1, padx=10, pady=10)

progress_text = tk.Text(root, height=8)
progress_text.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

root.mainloop()
