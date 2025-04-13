import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

def fetch_stock_data(ticker, days=365):
    """Fetch stock data with multiple fallback options"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Try Yahoo Finance first
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not df.empty:
            return df
    except Exception as e:
        print(f"Yahoo Finance error: {e}")
    
    # If Yahoo fails, try Alpha Vantage as fallback
    ALPHA_VANTAGE_API_KEY = 'NWD9PEU47GV2GB82'  # Get free key from https://www.alphavantage.co/
    if ALPHA_VANTAGE_API_KEY != 'NWD9PEU47GV2GB82':
        try:
            print(f"Trying Alpha Vantage for {ticker}...")
            if ticker.endswith('.NS'):
                symbol = f"NSE:{ticker.split('.')[0]}"
            elif ticker.endswith('.BO'):
                symbol = f"BOM:{ticker.split('.')[0]}"
            else:
                symbol = ticker
                
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
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
    
    # Final fallback to NSEpy (for Indian stocks only)
    try:
        print("Trying NSEpy...")
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
                return df
    except ImportError:
        print("NSEpy not installed. Install with: pip install nsepy")
    except Exception as e:
        print(f"NSEpy failed: {e}")
    
    return None

def analyze_reliable_indian_stock(ticker='RELIANCE.NS'):
    """Analyze stocks known to have good data availability"""
    # List of reliable Indian stocks with good data
    reliable_stocks = {
        'RELIANCE.NS': 'Reliance Industries (NSE)',
        'TATAMOTORS.NS': 'Tata Motors (NSE)',
        'HDFCBANK.NS': 'HDFC Bank (NSE)',
        'INFY.NS': 'Infosys (NSE)',
        'ICICIBANK.NS': 'ICICI Bank (NSE)'
    }
    
    if ticker not in reliable_stocks:
        print(f"\n⚠️ {ticker} not in reliable stocks list. Switching to RELIANCE.NS")
        ticker = 'RELIANCE.NS'
    
    print(f"\n🔍 Analyzing {reliable_stocks[ticker]} ({ticker})...")
    
    df = fetch_stock_data(ticker)
    
    if df is None or df.empty:
        print(f"❌ Could not fetch data for {ticker}")
        print("Trying alternative reliable stocks...")
        
        # Try other reliable stocks
        for alt_ticker, name in reliable_stocks.items():
            if alt_ticker != ticker:
                print(f"Attempting {name} ({alt_ticker})...")
                df = fetch_stock_data(alt_ticker)
                if df is not None and not df.empty:
                    ticker = alt_ticker
                    print(f"Success with {ticker}")
                    break
        
        if df is None or df.empty:
            print("❌ All attempts failed. Possible solutions:")
            print("1. Try again later (API might be temporarily unavailable)")
            print("2. Get a free Alpha Vantage API key")
            print("3. Install NSEpy for Indian market data (pip install nsepy)")
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
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Price and Moving Averages
    plt.subplot(2, 1, 1)
    plt.plot(df['Close'], label='Close Price', color='dodgerblue', linewidth=2)
    plt.plot(df['SMA20'], label='20-Day SMA', color='orange', linestyle='--')
    plt.plot(df['SMA50'], label='50-Day SMA', color='green', linestyle='--')
    plt.plot(df['SMA200'], label='200-Day SMA', color='red', linestyle='--')
    plt.title(f"{reliable_stocks[ticker]} Technical Analysis", fontweight='bold')
    plt.ylabel("Price (₹)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Indicators
    plt.subplot(2, 1, 2)
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
    
    print(f"\n📈 {reliable_stocks[ticker]} Technical Report as of {latest.name.date()}")
    print(f"Close Price: ₹{latest['Close']:.2f}")
    print(f"Daily Change: {((latest['Close']-prev_day['Close'])/prev_day['Close'])*100:.2f}%")
    print(f"Volume: {latest['Volume']:,.0f} shares")
    
    print("\n📊 Moving Averages:")
    print(f"20-Day SMA: ₹{latest['SMA20']:.2f} | Price is {'above' if latest['Close'] > latest['SMA20'] else 'below'}")
    print(f"50-Day SMA: ₹{latest['SMA50']:.2f} | Price is {'above' if latest['Close'] > latest['SMA50'] else 'below'}")
    print(f"200-Day SMA: ₹{latest['SMA200']:.2f} | Price is {'above' if latest['Close'] > latest['SMA200'] else 'below'}")
    
    print("\n📉 Technical Indicators:")
    print(f"RSI (14): {latest['RSI']:.2f} - {'Overbought (>70)' if latest['RSI'] > 70 else 'Oversold (<30)' if latest['RSI'] < 30 else 'Neutral'}")
    print(f"MACD: {latest['MACD']:.4f} - {'Bullish' if latest['MACD'] > 0 else 'Bearish'}")
    
    # Generate recommendation
    print("\n💡 Recommendation:")
    
    if latest['Close'] > latest['SMA200'] and latest['SMA50'] > latest['SMA200'] and latest['RSI'] < 70:
        print("✅ STRONG BUY - Uptrend with positive momentum")
    elif latest['Close'] < latest['SMA200'] and latest['SMA50'] < latest['SMA200'] and latest['RSI'] > 30:
        print("🚫 STRONG SELL - Downtrend with negative momentum")
    elif latest['RSI'] < 30 and latest['Close'] > latest['SMA200']:
        print("🟢 BUY - Oversold in long-term uptrend")
    elif latest['RSI'] > 70 and latest['Close'] < latest['SMA200']:
        print("🔴 SELL - Overbought in long-term downtrend")
    else:
        print("🟡 HOLD - Neutral or mixed signals")

# Example usage
if __name__ == "__main__":
    # Analyze one of the reliable Indian stocks
    analyze_reliable_indian_stock('RELIANCE.NS')  # Default - Reliance Industries
    
    # You can also try:
    # analyze_reliable_indian_stock('TATAMOTORS.NS')  # Tata Motors
    # analyze_reliable_indian_stock('HDFCBANK.NS')    # HDFC Bank
    # analyze_reliable_indian_stock('INFY.NS')        # Infosys