import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# -------------------------

# 0. Suppress convergence warnings

# -------------------------

warnings.simplefilter('ignore', ConvergenceWarning)

# -------------------------

# 1. Download BTC data

# -------------------------

ticker = 'ETH-USD'  # Use correct Yahoo ticker
start_date = '2024-01-01'
print(f"Loading data for {ticker} from {start_date}...")

df = yf.download(ticker, start=start_date, progress=False)[['Close']].rename(columns={'Close':'BTC'})
df.dropna(inplace=True)
print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")

# -------------------------

# 2. Log-transform

# -------------------------

y = df['BTC']
y_log = np.log(y)

# -------------------------

# 3. Split train/test (optional, here we fit on full data)

# -------------------------

train_size = int(len(y_log) * 0.9)
y_train_log = y_log.iloc[:train_size]
y_test_log = y_log.iloc[train_size:]

# -------------------------

# 4. Auto-ARIMA to find best SARIMA parameters

# -------------------------

print("Finding best SARIMA parameters via Auto-ARIMA...")
auto_model = pm.auto_arima(
y_train_log,
start_p=0, start_q=0,
max_p=3, max_q=3,
m=7,
d=None, D=None,
seasonal=True,
stepwise=True,
trace=True,
error_action='ignore',
suppress_warnings=True
)

best_params = auto_model.get_params()
order = best_params.get('order', (1,1,1))
seasonal_order = best_params.get('seasonal_order', (0,1,1,7))

print(f"Best parameters found: order={order}, seasonal_order={seasonal_order}")

# -------------------------

# 5. Fit SARIMA on full data

# -------------------------

print("Fitting SARIMA model on full data...")
model = SARIMAX(
y_log,
order=order,
seasonal_order=seasonal_order,
enforce_stationarity=False,
enforce_invertibility=False
)
model_fit = model.fit(disp=False, maxiter=1000)#, method='lbfgs'
print("Model fitted successfully.")

# -------------------------

# 6. Forecast next N days

# -------------------------

n_days = 10
forecast_log = model_fit.forecast(steps=n_days)
forecast = np.exp(forecast_log)

# Add future dates

last_date = y_log.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
forecast_series = pd.Series(forecast.values, index=future_dates)

# -------------------------

# 7. Plot historical + forecast

# -------------------------

plt.figure(figsize=(14,7))
plt.plot(y, label='Historical BTC', color='blue')
plt.plot(forecast_series, label=f'SARIMA {n_days}-day Forecast', color='red', linestyle='--')
plt.title(f"BTC Price Forecast ({n_days} days ahead)")
plt.xlabel("Date")
plt.ylabel("BTC Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------

# 8. Optional: show forecast table

# -------------------------

print("\nForecasted BTC Prices:")
print(forecast_series)
