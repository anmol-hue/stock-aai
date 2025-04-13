import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import numpy as np

def fetch_stock_data(ticker, days=180, retries=3):
    """Fetch stock data using yfinance with retry logic"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for attempt in range(retries):
        print(f"Attempting yfinance for {ticker} (Attempt {attempt+1}/{retries})...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, repair=True, auto_adjust=False)
            if not df.empty:
                print(f"yfinance succeeded for {ticker}")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            else:
                print(f"yfinance returned empty data for {ticker}")
        except Exception as e:
            print(f"yfinance error for {ticker}: {e}")
        time.sleep(2)
    
    print(f"Failed to fetch data for {ticker}")
    return None

def analyze_indian_stocks(tickers=['RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'ICICIBANK.NS', 'INFY.NS']):
    """Analyze Indian stocks with robust error handling and enhanced indicators"""
    results = {}
    best_stock = None
    best_score = -float('inf')
    
    for ticker in tickers:
        print(f"\n🔍 Analyzing {ticker}...")
        df = fetch_stock_data(ticker)
        
        if df is None or df.empty:
            print(f"❌ Skipping {ticker} due to data fetch failure")
            continue
        
        # Clean data
        df = df.dropna()
        if len(df) < 50:
            print(f"⚠️ Insufficient data for {ticker} ({len(df)} days)")
            continue
        
        # Calculate technical indicators
        try:
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
        except Exception as e:
            print(f"SMA calculation failed for {ticker}: {e}")
            df['SMA20'] = df['SMA50'] = np.nan
        
        try:
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        except Exception as e:
            print(f"RSI calculation failed for {ticker}: {e}")
            df['RSI'] = np.nan
        
        try:
            df['MACD'] = ta.trend.MACD(df['Close']).macd_diff()
        except Exception as e:
            print(f"MACD calculation failed for {ticker}: {e}")
            df['MACD'] = np.nan
        
        try:
            df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], window=20)
            df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], window=20)
        except Exception as e:
            print(f"Bollinger Bands calculation failed for {ticker}: {e}")
            df['BB_upper'] = df['BB_lower'] = np.nan
        
        try:
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        except Exception as e:
            print(f"ATR calculation failed for {ticker}: {e}")
            df['ATR'] = np.nan
        
        try:
            df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        except Exception as e:
            print(f"ADX calculation failed for {ticker}: {e}")
            df['ADX'] = np.nan
        
        try:
            df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
        except Exception as e:
            print(f"Stochastic calculation failed for {ticker}: {e}")
            df['Stoch'] = np.nan
        
        try:
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        except Exception as e:
            print(f"OBV calculation failed for {ticker}: {e}")
            df['OBV'] = np.nan
        
        # Score the stock
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
        
        results[ticker] = {
            'df': df,
            'score': score,
            'latest': latest,
            'volatility': volatility
        }
        
        if score > best_score:
            best_score = score
            best_stock = ticker
    
    # Print analysis for each stock
    summary_data = []
    for ticker, data in results.items():
        df = data['df']
        latest = data['latest']
        volatility = data['volatility']
        
        print(f"\n📈 {ticker} Analysis (as of {latest.name.date()})")
        print(f"Close: ₹{latest['Close']:.2f}")
        
        sma20 = f"₹{latest['SMA20']:.2f}" if pd.notna(latest['SMA20']) else 'N/A'
        sma20_status = 'above' if pd.notna(latest['SMA20']) and latest['Close'] > latest['SMA20'] else 'below' if pd.notna(latest['SMA20']) else 'N/A'
        print(f"20-Day SMA: {sma20} | Price is {sma20_status}")
        
        sma50 = f"₹{latest['SMA50']:.2f}" if pd.notna(latest['SMA50']) else 'N/A'
        sma50_status = 'above' if pd.notna(latest['SMA50']) and latest['Close'] > latest['SMA50'] else 'below' if pd.notna(latest['SMA50']) else 'N/A'
        print(f"50-Day SMA: {sma50} | Price is {sma50_status}")
        
        rsi = f"{latest['RSI']:.2f}" if pd.notna(latest['RSI']) else 'N/A'
        rsi_status = ('Overbought (>70)' if pd.notna(latest['RSI']) and latest['RSI'] > 70 else
                      'Oversold (<30)' if pd.notna(latest['RSI']) and latest['RSI'] < 30 else
                      'Neutral' if pd.notna(latest['RSI']) else 'N/A')
        print(f"RSI: {rsi} - {rsi_status}")
        
        macd = f"{latest['MACD']:.4f}" if pd.notna(latest['MACD']) else 'N/A'
        macd_status = 'Bullish' if pd.notna(latest['MACD']) and latest['MACD'] > 0 else 'Bearish' if pd.notna(latest['MACD']) else 'N/A'
        print(f"MACD: {macd} - {macd_status}")
        
        bb_lower = f"₹{latest['BB_lower']:.2f}" if pd.notna(latest['BB_lower']) else 'N/A'
        bb_upper = f"₹{latest['BB_upper']:.2f}" if pd.notna(latest['BB_upper']) else 'N/A'
        print(f"Bollinger Bands: {bb_lower} (Lower), {bb_upper} (Upper)")
        
        atr = f"₹{latest['ATR']:.2f}" if pd.notna(latest['ATR']) else 'N/A'
        print(f"ATR: {atr} (Volatility: {volatility*100:.1f}%)")
        
        adx = f"{latest['ADX']:.2f}" if pd.notna(latest['ADX']) else 'N/A'
        adx_status = ('Strong trend' if pd.notna(latest['ADX']) and latest['ADX'] > 25 else
                      'Weak trend' if pd.notna(latest['ADX']) and latest['ADX'] < 15 else
                      'Moderate trend' if pd.notna(latest['ADX']) else 'N/A')
        print(f"ADX: {adx} - {adx_status}")
        
        stoch = f"{latest['Stoch']:.2f}" if pd.notna(latest['Stoch']) else 'N/A'
        stoch_status = ('Oversold (<20)' if pd.notna(latest['Stoch']) and latest['Stoch'] < 20 else
                        'Overbought (>80)' if pd.notna(latest['Stoch']) and latest['Stoch'] > 80 else
                        'Neutral' if pd.notna(latest['Stoch']) else 'N/A')
        print(f"Stochastic: {stoch} - {stoch_status}")
        
        obv = f"{latest['OBV']:.0f}" if pd.notna(latest['OBV']) else 'N/A'
        print(f"OBV: {obv}")
        print(f"Score: {data['score']}")
        
        # Buy/Sell Signals
        buy_conditions = [
            pd.notna(latest['RSI']) and latest['RSI'] < 30,
            pd.notna(latest['BB_lower']) and latest['Close'] < latest['BB_lower'],
            pd.notna(latest['Stoch']) and latest['Stoch'] < 20,
            (pd.notna(latest['SMA20']) and pd.notna(latest['MACD']) and pd.notna(latest['ADX']) and
             latest['Close'] > latest['SMA20'] and latest['MACD'] > 0 and latest['ADX'] > 20)
        ]
        sell_conditions = [
            pd.notna(latest['RSI']) and latest['RSI'] > 70,
            pd.notna(latest['BB_upper']) and latest['Close'] > latest['BB_upper'],
            pd.notna(latest['Stoch']) and latest['Stoch'] > 80,
            (pd.notna(latest['SMA20']) and pd.notna(latest['MACD']) and pd.notna(latest['ADX']) and
             latest['Close'] < latest['SMA20'] and latest['MACD'] < 0 and latest['ADX'] > 20)
        ]
        print("\n💡 Recommendation:")
        if any(buy_conditions):
            print("✅ BUY - Oversold or strong bullish trend")
        elif any(sell_conditions):
            print("🚫 SELL - Overbought or bearish trend")
        else:
            print("🟡 HOLD - Neutral or mixed signals")
        
        # Trend Prediction
        print("📉 Trend Prediction (Short-Term):")
        if (pd.notna(latest['SMA20']) and pd.notna(latest['MACD']) and pd.notna(latest['ADX']) and
            latest['Close'] > latest['SMA20'] and latest['MACD'] > 0 and latest['ADX'] > 20):
            print("⬆️ Likely to go UP (strong bullish trend)")
        elif (pd.notna(latest['SMA20']) and pd.notna(latest['MACD']) and pd.notna(latest['ADX']) and
              latest['Close'] < latest['SMA20'] and latest['MACD'] < 0 and latest['ADX'] > 20):
            print("⬇️ Likely to go DOWN (strong bearish trend)")
        else:
            print("↔️ Uncertain or consolidating")
        
        # Plot
        try:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(df['Close'], label='Close Price', color='blue')
            if 'SMA20' in df:
                plt.plot(df['SMA20'], label='20-Day SMA', color='orange', linestyle='--')
            if 'BB_upper' in df:
                plt.plot(df['BB_upper'], label='Upper BB', color='red', linestyle=':')
            if 'BB_lower' in df:
                plt.plot(df['BB_lower'], label='Lower BB', color='green', linestyle=':')
            plt.title(f"{ticker} Technical Analysis")
            plt.ylabel("Price (₹)")
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.subplot(2, 1, 2)
            if 'RSI' in df:
                plt.plot(df['RSI'], label='RSI', color='purple')
                plt.axhline(70, color='red', linestyle='--', alpha=0.5)
                plt.axhline(30, color='green', linestyle='--', alpha=0.5)
                plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting failed for {ticker}: {e}")
        
        # Summary data
        summary_data.append({
            'Ticker': ticker,
            'Close': f"₹{latest['Close']:.2f}",
            'RSI': rsi,
            'MACD': macd,
            'ADX': adx,
            'Volatility': f"{volatility*100:.1f}%",
            'Score': data['score'],
            'Recommendation': 'BUY' if any(buy_conditions) else 'SELL' if any(sell_conditions) else 'HOLD'
        })
    
    # Summary table
    if results:
        print("\n📊 Summary of All Stocks")
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    
    # Best stock
    if best_stock:
        print(f"\n🏆 Best Stock: {best_stock} (Score: {results[best_stock]['score']})")
        latest = results[best_stock]['latest']
        print(f"Why: Strongest combination of trend, momentum, and stability")
        print(f"Current Price: ₹{latest['Close']:.2f}")
        print(f"Volatility: {results[best_stock]['volatility']*100:.1f}%")
        print(f"Recommendation: {'BUY' if (pd.isna(latest['RSI']) or latest['RSI'] < 70) and pd.notna(latest['MACD']) and latest['MACD'] > 0 else 'HOLD or evaluate further'}")
    else:
        print("\n❌ No stocks could be analyzed")
        print("Possible reasons:")
        print("- Data fetch or indicator calculation failed")
        print("- Update libraries: pip install yfinance pandas ta matplotlib --upgrade")
        print("- Check internet connection")

if __name__ == "__main__":
    analyze_indian_stocks()