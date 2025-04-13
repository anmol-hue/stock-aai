import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

def fetch_stock_data(ticker, exchange='BSE', days=365):
    """
    Fetch stock data with multiple fallback options
    For BSE: Use '.BO' suffix (e.g., 'BDL.BO')
    For NSE: Use '.NS' suffix (e.g., 'HDFCBANK.NS')
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Try Yahoo Finance first
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not df.empty:
            return df
    except:
        pass
    
    # If Yahoo fails, try alternative methods
    print(f"Yahoo Finance failed for {ticker}, trying alternative methods...")
    
    # Method 1: Try different ticker format
    if exchange == 'BSE' and not ticker.endswith('.BO'):
        new_ticker = f"{ticker.split('.')[0]}.BO"
        try:
            df = yf.download(new_ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                print(f"Success with {new_ticker}")
                return df
        except:
            pass
    
    # Method 2: Try Alpha Vantage as fallback (requires API key)
    ALPHA_VANTAGE_API_KEY = 'NWD9PEU47GV2GB82'  # Get free key from https://www.alphavantage.co/
    if ALPHA_VANTAGE_API_KEY != 'NWD9PEU47GV2GB82':
        try:
            if exchange == 'BSE':
                print("Trying Alpha Vantage (BSE)...")
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=BOM:{ticker.split('.')[0]}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
            else:
                print("Trying Alpha Vantage (NSE)...")
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NSE:{ticker.split('.')[0]}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
            
            response = requests.get(url)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                df = df.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                })
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df = df[start_date:end_date]
                df = df.apply(pd.to_numeric)
                return df
        except Exception as e:
            print(f"Alpha Vantage failed: {e}")
    
    print("All data fetch methods failed")
    return None

def analyze_stock(ticker, exchange='BSE'):
    print(f"\n🔍 Analyzing {ticker} ({exchange})...")
    
    # Add proper suffix if not present
    if exchange == 'BSE' and not ticker.endswith('.BO'):
        ticker = f"{ticker}.BO"
    elif exchange == 'NSE' and not ticker.endswith('.NS'):
        ticker = f"{ticker}.NS"
    
    df = fetch_stock_data(ticker, exchange)
    
    if df is None or df.empty:
        print(f"❌ Could not fetch data for {ticker}")
        print("Possible solutions:")
        print("1. Verify the ticker symbol is correct")
        print("2. For BSE stocks, ensure it ends with .BO (e.g., BDL.BO)")
        print("3. Try again later (API might be temporarily unavailable)")
        print("4. Consider getting an Alpha Vantage API key for more reliable data")
        return
    
    # Clean and prepare data
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    
    if len(df) < 50:
        print(f"⚠️ Only {len(df)} days of data available - need at least 50 for proper analysis")
        return
    
    # Calculate technical indicators
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd_diff()
    
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['Upper Band'] = bollinger.bollinger_hband()
    df['Lower Band'] = bollinger.bollinger_lband()
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Price and Moving Averages
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(df['Close'], label='Close Price', color='dodgerblue', linewidth=2)
    plt.plot(df['SMA20'], label='20-Day SMA', color='orange', linestyle='--')
    plt.plot(df['SMA50'], label='50-Day SMA', color='green', linestyle='--')
    plt.plot(df['SMA200'], label='200-Day SMA', color='red', linestyle='--')
    plt.fill_between(df.index, df['Upper Band'], df['Lower Band'], color='lightgray', alpha=0.3)
    plt.title(f"{ticker} Technical Analysis", fontweight='bold')
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Volume
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.bar(df.index, df['Volume'], color='royalblue', alpha=0.7)
    plt.title("Trading Volume")
    plt.ylabel("Volume")
    plt.grid(alpha=0.3)
    
    # Indicators
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', alpha=0.5)
    plt.axhline(30, color='green', linestyle='--', alpha=0.5)
    plt.plot(df['MACD'], label='MACD', color='orange')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.5)
    plt.title("Technical Indicators")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Generate analysis report
    latest = df.iloc[-1]
    prev_day = df.iloc[-2]
    
    print(f"\n📈 {ticker} Technical Report as of {latest.name.date()}")
    print(f"Close Price: ₹{latest['Close']:.2f}")
    print(f"Daily Change: {((latest['Close']-prev_day['Close'])/prev_day['Close'])*100:.2f}%")
    print(f"Volume: {latest['Volume']:,.0f} shares")
    
    print("\n📊 Key Levels:")
    print(f"20-Day SMA: ₹{latest['SMA20']:.2f} | Price is {'above' if latest['Close'] > latest['SMA20'] else 'below'}")
    print(f"50-Day SMA: ₹{latest['SMA50']:.2f} | Price is {'above' if latest['Close'] > latest['SMA50'] else 'below'}")
    print(f"200-Day SMA: ₹{latest['SMA200']:.2f} | Price is {'above' if latest['Close'] > latest['SMA200'] else 'below'}")
    print(f"Bollinger Bands: Upper ₹{latest['Upper Band']:.2f} | Lower ₹{latest['Lower Band']:.2f}")
    
    print("\n📉 Technical Indicators:")
    print(f"RSI (14): {latest['RSI']:.2f} - {'Overbought (>70)' if latest['RSI'] > 70 else 'Oversold (<30)' if latest['RSI'] < 30 else 'Neutral'}")
    print(f"MACD: {latest['MACD']:.4f} - {'Bullish' if latest['MACD'] > 0 else 'Bearish'}")
    
    # Trend analysis
    trend_strength = 0
    trend_text = []
    
    if latest['Close'] > latest['SMA200']:
        trend_strength += 1
        trend_text.append("Long-term uptrend (Price > 200SMA)")
    else:
        trend_strength -= 1
        trend_text.append("Long-term downtrend (Price < 200SMA)")
    
    if latest['SMA50'] > latest['SMA200']:
        trend_strength += 1
        trend_text.append("Golden Cross (50SMA > 200SMA)")
    else:
        trend_strength -= 1
        trend_text.append("Death Cross (50SMA < 200SMA)")
    
    if latest['Close'] > latest['SMA20']:
        trend_strength += 0.5
        trend_text.append("Short-term uptrend (Price > 20SMA)")
    else:
        trend_strength -= 0.5
        trend_text.append("Short-term downtrend (Price < 20SMA)")
    
    print("\n📌 Trend Analysis:")
    for text in trend_text:
        print(f"- {text}")
    
    # Generate recommendation
    print("\n💡 Recommendation:")
    
    if trend_strength >= 2 and latest['RSI'] < 70:
        print("✅ STRONG BUY - Strong uptrend with positive momentum")
    elif trend_strength <= -2 and latest['RSI'] > 30:
        print("🚫 STRONG SELL - Strong downtrend with negative momentum")
    elif trend_strength >= 1 and latest['RSI'] < 40:
        print("🟢 BUY - Uptrend with oversold conditions")
    elif trend_strength <= -1 and latest['RSI'] > 60:
        print("🔴 SELL - Downtrend with overbought conditions")
    elif latest['RSI'] < 30 and latest['Close'] > latest['SMA200']:
        print("🟢 CONSIDER BUYING - Oversold in long-term uptrend")
    elif latest['RSI'] > 70 and latest['Close'] < latest['SMA200']:
        print("🔴 CONSIDER SELLING - Overbought in long-term downtrend")
    else:
        print("🟡 HOLD / WATCH - Neutral or mixed signals")

# Example usage for BDL (Bharat Dynamics Limited) on BSE
if __name__ == "__main__":
    analyze_stock("BDL", exchange='BSE')  # Will automatically add .BO suffix
    
    # For NSE stocks:
    # analyze_stock("HDFCBANK", exchange='NSE')  # Will add .NS suffix