import time
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time as dt_time
import numpy as np

# Predefined stock tickers
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
            st.warning(f"yfinance error for {ticker}: {e}")
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
        st.warning(f"Indicator calculation failed for {ticker}: {e}")
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
    
    # General recommendation
    today = datetime.now().date()
    trading_start = dt_time(9, 15)  # NSE opens at 9:15 AM IST
    recommendation = {}
    
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
    
    # Predict best buy day
    buy_prediction = predict_best_buy_day(df, today)
    
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
        'trend': trend,
        'buy_prediction': buy_prediction
    }

def predict_best_buy_day(df, today):
    """Predict the best day to buy within the next 7 days"""
    last_5 = df.tail(5)
    latest = df.iloc[-1]
    
    rsi_trend = last_5['RSI'].diff().mean() if 'RSI' in last_5 else 0
    price_trend = last_5['Close'].diff().mean() if 'Close' in last_5 else 0
    macd_trend = last_5['MACD'].diff().mean() if 'MACD' in last_5 else 0
    
    days_ahead = 1
    for i in range(1, 8):
        future_date = today + timedelta(days=i)
        est_rsi = latest['RSI'] + rsi_trend * i if pd.notna(latest['RSI']) else 50
        est_price = latest['Close'] + price_trend * i
        est_macd = latest['MACD'] + macd_trend * i if pd.notna(latest['MACD']) else 0
        
        if (est_rsi < 30) or \
           (pd.notna(latest['BB_lower']) and est_price < latest['BB_lower']) or \
           (pd.notna(latest['SMA20']) and est_price < latest['SMA20'] and est_macd > 0):
            return {
                'when': f"{future_date.strftime('%Y-%m-%d')} at 9:15 AM IST",
                'reason': 'Predicted oversold or bullish crossover'
            }
        days_ahead = i + 1
    
    return {
        'when': f"{(today + timedelta(days=7)).strftime('%Y-%m-%d')} at 9:15 AM IST",
        'reason': 'No strong buy signal in next 7 days; monitor for opportunities'
    }

def display_stock_analysis(ticker, result):
    """Display analysis for a stock in Streamlit"""
    latest = result['latest']
    volatility = result['volatility']
    rec = result['recommendation']
    
    st.subheader(f"📈 {ticker} Analysis (as of {latest.name.date()})")
    st.write(f"**Close:** ₹{latest['Close']:.2f}")
    sma20 = f"₹{latest['SMA20']:.2f}" if pd.notna(latest['SMA20']) else 'N/A'
    st.write(f"**20-Day SMA:** {sma20} | Price is {'above' if pd.notna(latest['SMA20']) and latest['Close'] > latest['SMA20'] else 'below' if pd.notna(latest['SMA20']) else 'N/A'}")
    sma50 = f"₹{latest['SMA50']:.2f}" if pd.notna(latest['SMA50']) else 'N/A'
    st.write(f"**50-Day SMA:** {sma50} | Price is {'above' if pd.notna(latest['SMA50']) and latest['Close'] > latest['SMA50'] else 'below' if pd.notna(latest['SMA50']) else 'N/A'}")
    rsi = f"{latest['RSI']:.2f}" if pd.notna(latest['RSI']) else 'N/A'
    st.write(f"**RSI:** {rsi} - {'Overbought (>70)' if pd.notna(latest['RSI']) and latest['RSI'] > 70 else 'Oversold (<30)' if pd.notna(latest['RSI']) and latest['RSI'] < 30 else 'Neutral' if pd.notna(latest['RSI']) else 'N/A'}")
    macd = f"{latest['MACD']:.4f}" if pd.notna(latest['MACD']) else 'N/A'
    st.write(f"**MACD:** {macd} - {'Bullish' if pd.notna(latest['MACD']) and latest['MACD'] > 0 else 'Bearish' if pd.notna(latest['MACD']) else 'N/A'}")
    bb_lower = f"₹{latest['BB_lower']:.2f}" if pd.notna(latest['BB_lower']) else 'N/A'
    bb_upper = f"₹{latest['BB_upper']:.2f}" if pd.notna(latest['BB_upper']) else 'N/A'
    st.write(f"**Bollinger Bands:** {bb_lower} (Lower), {bb_upper} (Upper)")
    atr = f"₹{latest['ATR']:.2f}" if pd.notna(latest['ATR']) else 'N/A'
    st.write(f"**ATR:** {atr} (Volatility: {volatility*100:.1f}%)")
    adx = f"{latest['ADX']:.2f}" if pd.notna(latest['ADX']) else 'N/A'
    st.write(f"**ADX:** {adx} - {'Strong' if pd.notna(latest['ADX']) and latest['ADX'] > 25 else 'Weak' if pd.notna(latest['ADX']) and latest['ADX'] < 15 else 'Moderate' if pd.notna(latest['ADX']) else 'N/A'}")
    stoch = f"{latest['Stoch']:.2f}" if pd.notna(latest['Stoch']) else 'N/A'
    st.write(f"**Stochastic:** {stoch} - {'Oversold (<20)' if pd.notna(latest['Stoch']) and latest['Stoch'] < 20 else 'Overbought (>80)' if pd.notna(latest['Stoch']) and latest['Stoch'] > 80 else 'Neutral' if pd.notna(latest['Stoch']) else 'N/A'}")
    obv = f"{latest['OBV']:.0f}" if pd.notna(latest['OBV']) else 'N/A'
    st.write(f"**OBV:** {obv}")
    st.write(f"**Score:** {result['score']}")
    
    st.write("### 💡 Recommendation")
    st.write(f"**{rec['action']}** on {rec['when']} - {rec['reason']}")
    
    st.write("### 📉 Trend Prediction (Short-Term)")
    trend_icon = "⬆️ UP" if result['trend'] == 'UP' else "⬇️ DOWN" if result['trend'] == 'DOWN' else "↔️ Uncertain"
    st.write(trend_icon)
    
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
    st.pyplot(fig)

def create_summary_row(ticker, result):
    """Create a row for the summary table"""
    latest = result['latest']
    volatility = result['volatility']
    rec = result['recommendation']
    return {
        'Ticker': ticker,
        'Close': f"₹{latest['Close']:.2f}",
        'RSI': f"{latest['RSI']:.2f}" if pd.notna(latest['RSI']) else 'N/A',
        'MACD': f"{latest['MACD']:.4f}" if pd.notna(latest['MACD']) else 'N/A',
        'ADX': f"{latest['ADX']:.2f}" if pd.notna(latest['ADX']) else 'N/A',
        'Volatility': f"{volatility*100:.1f}%",
        'Score': result['score'],
        'Recommendation': f"{rec['action']} on {rec['when']}"
    }

# Streamlit UI
st.title("Stock Analysis Dashboard")

# Sidebar for inputs
st.sidebar.header("Analysis Settings")
tickers = st.sidebar.multiselect("Select Stocks to Compare", TICKERS, default=['RELIANCE.NS'])
search_ticker = st.sidebar.text_input("Search for a Stock (e.g., TATAMOTORS.NS)", "")
period_type = st.sidebar.radio("Period Type", ['Days', 'Months'])
period_value = st.sidebar.number_input(f"Number of {period_type}", min_value=1, value=180 if period_type == 'Days' else 6, step=1)

# Convert period to days
days = period_value if period_type == 'Days' else period_value * 30
if days < 50:
    days = 50
    st.sidebar.warning("Minimum period set to 50 days for reliable analysis.")

if st.sidebar.button("Run Analysis"):
    results = {}
    summary_data = []
    best_stock = None
    best_score = -float('inf')
    
    # Analyze selected stocks
    for ticker in tickers:
        with st.spinner(f"Analyzing {ticker}..."):
            result = analyze_stock(ticker, days)
            if result:
                results[ticker] = result
                display_stock_analysis(ticker, result)
                summary_data.append(create_summary_row(ticker, result))
                if result['score'] > best_score:
                    best_score = result['score']
                    best_stock = ticker
    
    # Analyze searched stock
    if search_ticker:
        with st.spinner(f"Analyzing {search_ticker}..."):
            search_result = analyze_stock(search_ticker, days)
            if search_result:
                st.subheader(f"🔍 Search Result: {search_ticker}")
                display_stock_analysis(search_ticker, search_result)
                st.write("### Best Time to Buy (if not currently recommended)")
                st.write(f"**Buy on:** {search_result['buy_prediction']['when']}")
                st.write(f"**Reason:** {search_result['buy_prediction']['reason']}")
            else:
                st.error(f"Could not analyze {search_ticker}. Verify ticker or try again.")
    
    # Display summary and best stock
    if results:
        st.subheader("📊 Summary of Selected Stocks")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
        
        if best_stock:
            st.subheader(f"🏆 Best Stock to Buy Now: {best_stock}")
            st.write(f"**Score:** {results[best_stock]['score']}")
            st.write(f"**Current Price:** ₹{results[best_stock]['latest']['Close']:.2f}")
            st.write(f"**Volatility:** {results[best_stock]['volatility']*100:.1f}%")
            st.write(f"**Recommendation:** {results[best_stock]['recommendation']['action']} on {results[best_stock]['recommendation']['when']}")
    else:
        st.error("No stocks could be analyzed. Check data availability or try again.")

# Instructions
st.sidebar.info("Select stocks to compare and/or search for a specific stock. Adjust the analysis period and click 'Run Analysis'.")