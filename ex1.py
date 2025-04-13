from nsepy import get_history
from datetime import date
df = get_history(symbol='INFY', start=date(2024, 10, 11), end=date(2025, 4, 11))
print(df.head())