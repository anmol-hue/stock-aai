import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime, timedelta, time as dt_time
import numpy as np

# Stock tickers
TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'ICICIBANK.NS', 'INFY.NS']

def fetch_stock_data(ticker, days=180, retries=3):
    """Fetch stock data using yfinance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, repair=True, auto_adjust=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            else:
                return None
        except Exception as e:
            print(f"yfinance error for {ticker}: {e}")
        time.sleep(2)
    return None

def analyze_stock(ticker, days):
    """Analyze a single stock and return results"""
    df = fetch_stock_data(ticker, days)
    if df is None or df.empty or len(df) < 50:
        return None
    
    # Calculate indicators
    try:
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.MACD(df['Close']).macd_diff()
        df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], window=20)
        df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], window=20)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    except Exception as e:
        print(f"Indicator calculation failed for {ticker}: {e}")
        return None
    
    # Score and recommendations
    latest = df.iloc[-1]
    score = 0
    volatility = latest['ATR'] / latest['Close'] if pd.notna(latest['ATR']) else 0.03
    
    if pd.notna(latest['SMA20']) and latest['Close'] > latest['SMA20']:
        score += 2
    if pd.notna(latest['SMA50']) and latest['Close'] > latest['SMA50']:
        score += 1
    if pd.notna(latest['BB_lower']) and latest['Close'] < latest['BB_lower']:
        score += 2
    elif pd.notna(latest['BB_upper']) and latest['Close'] > latest['BB_upper']:
        score -= 2
    if pd.notna(latest['RSI']):
        if 30 < latest['RSI'] < 70:
            score += 2
        elif latest['RSI'] < 30:
            score += 3
        elif latest['RSI'] > 70:
            score -= 2
    if pd.notna(latest['MACD']) and latest['MACD'] > 0:
        score += 2
    if pd.notna(latest['Stoch']):
        if latest['Stoch'] < 20:
            score += 2
        elif latest['Stoch'] > 80:
            score -= 2
    if pd.notna(latest['ADX']):
        if latest['ADX'] > 25:
            score += 2
        elif latest['ADX'] < 15:
            score -= 1
    if pd.notna(latest['OBV']) and len(df) > 1 and latest['OBV'] > df['OBV'].iloc[-2] and latest['Close'] > df['Close'].iloc[-2]:
        score += 1
    if volatility < 0.02:
        score += 1
    elif volatility > 0.05:
        score -= 1
    
    # Detailed recommendation with day/time
    last_5 = df.tail(5)
    recommendation = {}
    today = datetime.now().date()
    trading_start = dt_time(9, 15)  # NSE opens at 9:15 AM IST
    
    # Buy: Look for RSI uptrend, price crossing SMA20, or MACD crossover
    if (pd.notna(latest['RSI']) and latest['RSI'] < 30) or \
       (pd.notna(latest['Stoch']) and latest['Stoch'] < 20) or \
       (pd.notna(latest['BB_lower']) and latest['Close'] < latest['BB_lower']):
        recommendation['action'] = 'BUY'
        recommendation['when'] = f"{(today + timedelta(days=1)).strftime('%Y-%m-%d')} at 9:15 AM IST"
        recommendation['reason'] = 'Oversold conditions suggest a rebound'
    elif (pd.notna(latest['SMA20']) and latest['Close'] > latest['SMA20'] and
          pd.notna(latest['MACD']) and latest['MACD'] > 0 and pd.notna(latest['ADX']) and latest['ADX'] > 20):
        recommendation['action'] = 'BUY'
        recommendation['when'] = f"{(today + timedelta(days=1)).strftime('%Y-%m-%d')} at 9:15 AM IST"
        recommendation['reason'] = 'Strong bullish trend confirmed'
    # Sell: Look for RSI downturn, price crossing BB_upper, or MACD bearish
    elif (pd.notna(latest['RSI']) and latest['RSI'] > 70) or \
         (pd.notna(latest['Stoch']) and latest['Stoch'] > 80) or \
         (pd.notna(latest['BB_upper']) and latest['Close'] > latest['BB_upper']):
        recommendation['action'] = 'SELL'
        recommendation['when'] = f"{(today + timedelta(days=1)).strftime('%Y-%m-%d')} at 9:15 AM IST"
        recommendation['reason'] = 'Overbought conditions suggest a pullback'
    elif (pd.notna(latest['SMA20']) and latest['Close'] < latest['SMA20'] and
          pd.notna(latest['MACD']) and latest['MACD'] < 0 and pd.notna(latest['ADX']) and latest['ADX'] > 20):
        recommendation['action'] = 'SELL'
        recommendation['when'] = f"{(today + timedelta(days=1)).strftime('%Y-%m-%d')} at 9:15 AM IST"
        recommendation['reason'] = 'Strong bearish trend confirmed'
    else:
        recommendation['action'] = 'HOLD'
        recommendation['when'] = f"{today.strftime('%Y-%m-%d')} at any time"
        recommendation['reason'] = 'Neutral or consolidating signals'
    
    # Trend prediction
    trend = 'Uncertain'
    if (pd.notna(latest['SMA20']) and pd.notna(latest['MACD']) and pd.notna(latest['ADX']) and
        latest['Close'] > latest['SMA20'] and latest['MACD'] > 0 and latest['ADX'] > 20):
        trend = 'UP'
    elif (pd.notna(latest['SMA20']) and pd.notna(latest['MACD']) and pd.notna(latest['ADX']) and
          latest['Close'] < latest['SMA20'] and latest['MACD'] < 0 and latest['ADX'] > 20):
        trend = 'DOWN'
    
    return {
        'df': df,
        'score': score,
        'latest': latest,
        'volatility': volatility,
        'recommendation': recommendation,
        'trend': trend
    }

def run_analysis():
    """Run analysis based on UI inputs"""
    ticker = ticker_var.get()
    period_type = period_type_var.get()
    period_value = int(period_value_var.get())
    
    # Convert period to days
    days = period_value if period_type == 'Days' else period_value * 30
    if days < 50:
        days = 50  # Minimum for indicators
    
    result = analyze_stock(ticker, days)
    output_text.delete(1.0, tk.END)
    
    if result:
        latest = result['latest']
        volatility = result['volatility']
        rec = result['recommendation']
        
        output = f"📈 {ticker} Analysis (as of {latest.name.date()})\n"
        output += f"Close: ₹{latest['Close']:.2f}\n"
        sma20 = f"₹{latest['SMA20']:.2f}" if pd.notna(latest['SMA20']) else 'N/A'
        output += f"20-Day SMA: {sma20} | Price is {'above' if pd.notna(latest['SMA20']) and latest['Close'] > latest['SMA20'] else 'below' if pd.notna(latest['SMA20']) else 'N/A'}\n"
        sma50 = f"₹{latest['SMA50']:.2f}" if pd.notna(latest['SMA50']) else 'N/A'
        output += f"50-Day SMA: {sma50} | Price is {'above' if pd.notna(latest['SMA50']) and latest['Close'] > latest['SMA50'] else 'below' if pd.notna(latest['SMA50']) else 'N/A'}\n"
        rsi = f"{latest['RSI']:.2f}" if pd.notna(latest['RSI']) else 'N/A'
        output += f"RSI: {rsi} - {'Overbought (>70)' if pd.notna(latest['RSI']) and latest['RSI'] > 70 else 'Oversold (<30)' if pd.notna(latest['RSI']) and latest['RSI'] < 30 else 'Neutral' if pd.notna(latest['RSI']) else 'N/A'}\n"
        macd = f"{latest['MACD']:.4f}" if pd.notna(latest['MACD']) else 'N/A'
        output += f"MACD: {macd} - {'Bullish' if pd.notna(latest['MACD']) and latest['MACD'] > 0 else 'Bearish' if pd.notna(latest['MACD']) else 'N/A'}\n"
        bb_lower = f"₹{latest['BB_lower']:.2f}" if pd.notna(latest['BB_lower']) else 'N/A'
        bb_upper = f"₹{latest['BB_upper']:.2f}" if pd.notna(latest['BB_upper']) else 'N/A'
        output += f"Bollinger Bands: {bb_lower} (Lower), {bb_upper} (Upper)\n"
        atr = f"₹{latest['ATR']:.2f}" if pd.notna(latest['ATR']) else 'N/A'
        output += f"ATR: {atr} (Volatility: {volatility*100:.1f}%)\n"
        adx = f"{latest['ADX']:.2f}" if pd.notna(latest['ADX']) else 'N/A'
        output += f"ADX: {adx} - {'Strong' if pd.notna(latest['ADX']) and latest['ADX'] > 25 else 'Weak' if pd.notna(latest['ADX']) and latest['ADX'] < 15 else 'Moderate' if pd.notna(latest['ADX']) else 'N/A'}\n"
        stoch = f"{latest['Stoch']:.2f}" if pd.notna(latest['Stoch']) else 'N/A'
        output += f"Stochastic: {stoch} - {'Oversold (<20)' if pd.notna(latest['Stoch']) and latest['Stoch'] < 20 else 'Overbought (>80)' if pd.notna(latest['Stoch']) and latest['Stoch'] > 80 else 'Neutral' if pd.notna(latest['Stoch']) else 'N/A'}\n"
        obv = f"{latest['OBV']:.0f}" if pd.notna(latest['OBV']) else 'N/A'
        output += f"OBV: {obv}\n"
        output += f"Score: {result['score']}\n"
        
        output += "\n💡 Recommendation:\n"
        output += f"{rec['action']} on {rec['when']} - {rec['reason']}\n"
        
        output += "\n📉 Trend Prediction (Short-Term):\n"
        output += f"{'⬆️ UP' if result['trend'] == 'UP' else '⬇️ DOWN' if result['trend'] == 'DOWN' else '↔️ Uncertain'}\n"
        
        output_text.insert(tk.END, output)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(result['df']['Close'], label='Close Price', color='blue')
        ax1.plot(result['df']['SMA20'], label='20-Day SMA', color='orange', linestyle='--')
        ax1.plot(result['df']['BB_upper'], label='Upper BB', color='red', linestyle=':')
        ax1.plot(result['df']['BB_lower'], label='Lower BB', color='green', linestyle=':')
        ax1.set_title(f"{ticker} Technical Analysis")
        ax1.set_ylabel("Price (₹)")
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        if 'RSI' in result['df']:
            ax2.plot(result['df']['RSI'], label='RSI', color='purple')
            ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
            ax2.set_ylabel("RSI")
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    else:
        output_text.insert(tk.END, f"❌ Analysis failed for {ticker}. Check data or try again.\n")

# UI Setup
root = tk.Tk()
root.title("Stock Analysis Tool")
root.geometry("800x600")

# Ticker selection
tk.Label(root, text="Select Stock:").pack(pady=5)
ticker_var = tk.StringVar(value=TICKERS[0])
ticker_menu = ttk.Combobox(root, textvariable=ticker_var, values=TICKERS)
ticker_menu.pack(pady=5)

# Period selection
tk.Label(root, text="Analysis Period:").pack(pady=5)
period_frame = tk.Frame(root)
period_frame.pack(pady=5)

period_type_var = tk.StringVar(value='Days')
ttk.OptionMenu(period_frame, period_type_var, 'Days', 'Days', 'Months').pack(side=tk.LEFT, padx=5)
period_value_var = tk.StringVar(value='180')
period_entry = ttk.Entry(period_frame, textvariable=period_value_var, width=5)
period_entry.pack(side=tk.LEFT, padx=5)
tk.Label(period_frame, text="(Enter number)").pack(side=tk.LEFT)

# Run button
ttk.Button(root, text="Run Analysis", command=run_analysis).pack(pady=10)

# Output text
output_text = scrolledtext.ScrolledText(root, height=15, width=80)
output_text.pack(pady=5, fill=tk.BOTH, expand=True)

# Plot frame
plot_frame = tk.Frame(root)
plot_frame.pack(pady=5, fill=tk.BOTH, expand=True)

root.mainloop()