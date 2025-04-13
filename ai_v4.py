import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import time

def fetch_stock_data(ticker, days=180, retries=3):
    """Fetch stock data with multiple fallbacks and retry logic"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Try Yahoo Finance
    for attempt in range(retries):
        print(f"Attempting Yahoo Finance for {ticker} (Attempt {attempt+1}/{retries})...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, repair=True, progress=False)
            if not df.empty:
                print(f"Yahoo Finance succeeded for {ticker}")
                return df
            else:
                print(f"Yahoo Finance returned empty data for {ticker}")
        except Exception as e:
            print(f"Yahoo Finance error for {ticker}: {e}")
        time.sleep(2)  # Wait before retrying
    
    # Fallback to Alpha Vantage (requires API key)
    ALPHA_VANTAGE_API_KEY = 'NWD9PEU47GV2GB82'  # Replace with your Alpha Vantage key
    if ALPHA_VANTAGE_API_KEY != 'NWD9PEU47GV2GB82':
        print(f"Trying Alpha Vantage for {ticker}...")
        try:
            symbol = f"NSE:{ticker.split('.')[0]}" if ticker.endswith('.NS') else ticker
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url)
            data = response.json()
            if 'Time Series (Daily)' in data:
                print(f"Alpha Vantage succeeded for {ticker}")
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                df = df.rename(columns={
                    '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
                    '4. close': 'Close', '5. volume': 'Volume'
                })
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df = df[start_date:end_date]
                df = df.apply(pd.to_numeric)
                return df
            else:
                print(f"Alpha Vantage error for {ticker}: {data.get('Note', 'No data')}")
        except Exception as e:
            print(f"Alpha Vantage failed for {ticker}: {e}")
    
    # Fallback to NSEpy (for Indian stocks)
    try:
        print(f"Trying NSEpy for {ticker}...")
        from nsepy import get_history
        from datetime import date
        if ticker.endswith('.NS'):
            symbol = ticker.split('.')[0]
            df = get_history(
                symbol=symbol,
                start=date(start_date.year, start_date.month, start_date.day),
                end=date(end_date.year, end_date.month, end_date.day)
            )
            if not df.empty:
                print(f"NSEpy succeeded for {ticker}")
                return df
            else:
                print(f"NSEpy returned empty data for {ticker}")
    except ImportError:
        print("NSEpy not installed. Install with: pip install nsepy")
    except Exception as e:
        print(f"NSEpy failed for {ticker}: {e}")
    
    print(f"All data sources failed for {ticker}")
    return None

def analyze_indian_stocks(tickers=['RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'INFY.NS', 'ICICIBANK.NS']):
    """Analyze multiple Indian stocks and recommend the best based on technicals"""
    results = {}
    best_stock = None
    best_score = -float('inf')
    
    for ticker in tickers:
        print(f"\n🔍 Analyzing {ticker}...")
        df = fetch_stock_data(ticker)
        
        if df is None or df.empty:
            print(f"❌ Failed to fetch data for {ticker}")
            continue
        
        # Clean and prepare data
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        if len(df) < 50:
            print(f"⚠️ Insufficient data for {ticker} ({len(df)} days)")
            continue
        
        # Calculate technical indicators
        df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd_diff()
        
        # Score the stock based on indicators
        latest = df.iloc[-1]
        score = 0
        
        # Trend: Price above SMAs (bullish)
        if latest['Close'] > latest['SMA20']:
            score += 1
        if latest['Close'] > latest['SMA50']:
            score += 1
        if latest['SMA20'] > latest['SMA50']:
            score += 1  # Golden cross potential
        
        # Momentum: RSI in healthy range
        if 30 < latest['RSI'] < 70:
            score += 1
        elif latest['RSI'] < 30:
            score += 2  # Oversold (buy opportunity)
        elif latest['RSI'] > 70:
            score -= 2  # Overbought (sell risk)
        
        # MACD: Bullish signal
        if latest['MACD'] > 0:
            score += 1
        
        results[ticker] = {
            'df': df,
            'score': score,
            'latest': latest
        }
        
        # Update best stock
        if score > best_score:
            best_score = score
            best_stock = ticker
    
    # Print analysis for each stock
    for ticker, data in results.items():
        latest = data['latest']
        print(f"\n📈 {ticker} Analysis (as of {latest.name.date()})")
        print(f"Close: ₹{latest['Close']:.2f}")
        print(f"20-Day SMA: ₹{latest['SMA20']:.2f} | Price is {'above' if latest['Close'] > latest['SMA20'] else 'below'}")
        print(f"50-Day SMA: ₹{latest['SMA50']:.2f} | Price is {'above' if latest['Close'] > latest['SMA50'] else 'below'}")
        print(f"RSI: {latest['RSI']:.2f} - {'Overbought (>70)' if latest['RSI'] > 70 else 'Oversold (<30)' if latest['RSI'] < 30 else 'Neutral'}")
        print(f"MACD: {latest['MACD']:.4f} - {'Bullish' if latest['MACD'] > 0 else 'Bearish'}")
        
        # Buy/Sell Signals
        print("\n💡 Signals:")
        if latest['Close'] > latest['SMA20'] and latest['RSI'] < 70 and latest['MACD'] > 0:
            print("✅ BUY - Price above SMA20, RSI healthy, MACD bullish")
        elif latest['RSI'] < 30:
            print("🟢 BUY - Oversold (RSI < 30)")
        elif latest['Close'] < latest['SMA20'] and latest['RSI'] > 70 and latest['MACD'] < 0:
            print("🚫 SELL - Price below SMA20, RSI overbought, MACD bearish")
        elif latest['RSI'] > 70:
            print("🔴 SELL - Overbought (RSI > 70)")
        else:
            print("🟡 HOLD - Mixed or neutral signals")
        
        # Trend Prediction
        print("📉 Trend Prediction:")
        if latest['Close'] > latest['SMA20'] and latest['MACD'] > 0:
            print("⬆️ Likely to go UP (short-term bullish)")
        elif latest['Close'] < latest['SMA20'] and latest['MACD'] < 0:
            print("⬇️ Likely to go DOWN (short-term bearish)")
        else:
            print("↔️ Uncertain (sideways or consolidating)")
    
    # Plot the best stock
    if best_stock:
        print(f"\n🏆 Best Stock: {best_stock} (Score: {results[best_stock]['score']})")
        df = results[best_stock]['df']
        plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Close Price', color='dodgerblue')
        plt.plot(df['SMA20'], label='20-Day SMA', color='orange', linestyle='--')
        plt.plot(df['SMA50'], label='50-Day SMA', color='green', linestyle='--')
        plt.title(f"{best_stock} Technical Analysis")
        plt.ylabel("Price (₹)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    else:
        print("\n❌ No suitable stocks found for analysis")
        print("Suggestions:")
        print("- Ensure internet connectivity")
        print("- Update yfinance: pip install yfinance --upgrade")
        print("- Get an Alpha Vantage API key: https://www.alphavantage.co/")
        print("- Install nsepy: pip install nsepy")
    
    return best_stock

if __name__ == "__main__":
    # Analyze major Indian stocks
    analyze_indian_stocks()