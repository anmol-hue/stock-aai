import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import requests

def fetch_stock_data(ticker, days=365):
    """Fetch stock data with robust error handling"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Try both methods of downloading data
        try:
            # Method 1: Direct download
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        except:
            # Method 2: Using Ticker object if download fails
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            
        if df.empty:
            raise ValueError("Empty DataFrame returned")
            
        return df
    
    except json.JSONDecodeError as je:
        print(f"JSON decode error occurred. This might be a temporary Yahoo Finance API issue.")
        return None
    except requests.exceptions.RequestException as re:
        print(f"Network error occurred: {re}")
        return None
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def analyze_stock(ticker):
    print(f"\n🔍 Analyzing {ticker}...")
    
    # Fetch data with error handling
    df = fetch_stock_data(ticker)
    if df is None or df.empty:
        print(f"⚠️ Failed to fetch data for {ticker}. Trying alternative solutions...")
        
        # Try without .NS suffix for Indian stocks if first attempt fails
        if ticker.endswith('.NS'):
            print("Attempting without .NS suffix...")
            base_ticker = ticker[:-3]
            df = fetch_stock_data(base_ticker)
            
        if df is None or df.empty:
            print("❌ All attempts failed. Possible reasons:")
            print("- Invalid ticker symbol")
            print("- Yahoo Finance API issues (try again later)")
            print("- No data available for this ticker")
            print("Tip: For Indian stocks, try appending .NS (e.g., 'HDFCBANK.NS')")
            return
    
    # Clean data
    df.dropna(inplace=True)
    if len(df) < 50:  # Need at least 50 days for meaningful analysis
        print(f"⚠️ Insufficient data points ({len(df)}). Need at least 50 days of data.")
        return

    # Calculate Indicators
    try:
        df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        macd_calc = ta.trend.MACD(df['Close'])
        df['MACD'] = macd_calc.macd_diff()
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return

    # Plotting
    plt.figure(figsize=(14, 8))
    
    # Price and Moving Averages
    plt.subplot(2, 1, 1)
    plt.plot(df['Close'], label='Close Price', color='blue', alpha=0.75)
    plt.plot(df['SMA50'], label='SMA50', color='green', linestyle='--')
    plt.plot(df['SMA200'], label='SMA200', color='red', linestyle='--')
    plt.title(f"{ticker} Technical Analysis")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Indicators
    plt.subplot(2, 1, 2)
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', alpha=0.5)
    plt.axhline(30, color='green', linestyle='--', alpha=0.5)
    plt.plot(df['MACD'], label='MACD', color='orange')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.5)
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Analysis
    latest = df.iloc[-1]
    print(f"\n📊 Technical Analysis for {ticker} as of {latest.name.date()}")
    print(f"Close Price: ₹{latest['Close']:.2f}")
    print(f"50-Day SMA: ₹{latest['SMA50']:.2f}")
    print(f"200-Day SMA: ₹{latest['SMA200']:.2f}")
    print(f"RSI (14): {latest['RSI']:.2f}")
    print(f"MACD: {latest['MACD']:.4f}")

    # Trend analysis
    price_vs_sma50 = "above" if latest['Close'] > latest['SMA50'] else "below"
    price_vs_sma200 = "above" if latest['Close'] > latest['SMA200'] else "below"
    sma_cross = "Golden Cross (Bullish)" if latest['SMA50'] > latest['SMA200'] else "Death Cross (Bearish)"
    
    print(f"\n📈 Trend Analysis:")
    print(f"- Price is {price_vs_sma50} 50-Day SMA and {price_vs_sma200} 200-Day SMA")
    print(f"- Moving Averages: {sma_cross}")

    # RSI analysis
    rsi_signal = ""
    if latest['RSI'] < 30:
        rsi_signal = "🟢 Oversold (Potential Buy)"
    elif latest['RSI'] > 70:
        rsi_signal = "🔴 Overbought (Potential Sell)"
    else:
        rsi_signal = "⚪ Neutral"
    print(f"- RSI Signal: {rsi_signal}")

    # MACD analysis
    macd_signal = ""
    if latest['MACD'] > 0:
        macd_signal = "🟢 Bullish (Above zero line)"
    else:
        macd_signal = "🔴 Bearish (Below zero line)"
    print(f"- MACD Signal: {macd_signal}")

    # Recommendation
    print("\n💡 Recommendation:")
    if latest['SMA50'] > latest['SMA200'] and latest['RSI'] < 70 and latest['MACD'] > 0:
        print("✅ Strong Buy Signal - Uptrend with positive momentum")
    elif latest['SMA50'] < latest['SMA200'] and latest['RSI'] > 30 and latest['MACD'] < 0:
        print("🚫 Strong Sell Signal - Downtrend with negative momentum")
    elif latest['RSI'] < 30 and latest['Close'] > latest['SMA200']:
        print("🟢 Consider Buying - Oversold in long-term uptrend")
    elif latest['RSI'] > 70 and latest['Close'] < latest['SMA200']:
        print("🔴 Consider Selling - Overbought in long-term downtrend")
    else:
        print("🟡 Hold / Watch - Mixed signals or neutral market")

# Example usage
if __name__ == "__main__":
    ticker = "HDFCBANK.NS"  # For Indian stocks
    # ticker = "HDFCBANK.BO"  # Alternative for BSE
    # ticker = "RELIANCE.NS"  # Another Indian stock example
    
    # First verify the ticker exists
    try:
        stock_info = yf.Ticker(ticker).info
        if not stock_info:
            print(f"Ticker {ticker} not found. Trying alternatives...")
            ticker = "RELIANCE.NS"  # Try BSE instead of NSE
    except:
        pass
    
    analyze_stock(ticker)