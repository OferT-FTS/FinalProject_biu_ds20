import yfinance as yf
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# הורדת הנתונים
data = yf.download("SPY", start="2010-01-01", end="2025-08-30")['Close']
data_pch = data.pct_change().dropna()
# פירוק למרכיבים (שינוי בהתאם לתוצאות מבחן ADF)
result = None
adfuller_result = adfuller(data_pch)
if adfuller_result[1] <= 0.05:
    print("The time series is stationary.")
    result = seasonal_decompose(data_pch, model='additive', period=252)
else:
    print("The time series is not stationary. Applying differencing...")
    data_diff = data.diff().dropna() # Use pandas built-in diff and dropna
    result = seasonal_decompose(data_pch, model='additive', period=252)

# הצגת גרפים
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(data_pch, label='Original')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(result.trend, label='Trend')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(result.seasonal, label='Seasonal')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(result.resid, label='Residual')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# מבחן ADF על הנתונים לאחר הדיפרנציאציה (אם בוצעה)
if 'data_diff' in locals():
    result_adf = adfuller(data_pch)
    print(f'ADF Statistic (after differencing): {result_adf[0]}')
    print(f'p-value (after differencing): {result_adf[1]}')
    print('Critical Values:')
    for key, value in result_adf[4].items():
        print(f'   {key}, {value}')

    if result_adf[1] <= 0.05:
        print("The time series is stationary after differencing.")
    else:
        print("The time series is still not stationary after differencing.")
else:
    print("No differencing was applied as the original data was stationary.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm
import matplotlib.colors as mcolors

# הורדת נתונים מ-Yahoo Finance
data = yf.download("IBM", start="2020-01-01", end="2025-10-12")

# ניקוי נתונים
data['Adj Close'] = data['Close'].ffill()

# פונקציה להרצת חיזוי רולינג קדימה
def rolling_auto_arima(train_size_ratio=0.9):
    series = data['Adj Close']
    train_size = int(len(series) * train_size_ratio)
    train, test = series[:train_size], series[train_size:]
    history = list(train)
    predictions = []

    print(f"Training samples: {len(train)}, Test samples: {len(test)}")

    # יצירת רשימת צבעים ייחודית לכל חיזוי
    colors = list(mcolors.TABLEAU_COLORS.values())

    # חיזוי יום קדימה בכל פעם
    for i in range(len(test)):
        model = pm.auto_arima(
            history,
            start_p=0, max_p=4,
            start_q=0, max_q=4,
            start_d=0, max_d=2,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )

        forecast = model.predict(n_periods=1)[0]
        predictions.append(forecast)
        history.append(test.iloc[i])  # מוסיף את הערך האמיתי להיסטוריה

        print(f"Day {i+1}/{len(test)} | Predicted: {forecast:.2f} | Actual: {test.iloc[i]:.2f}")

    # מדדים
    mse = mean_squared_error(test, predictions)
    mae = mean_absolute_error(test, predictions)
    print(f"\nMSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    # גרף: חיזוי יומי בצבע שונה
    plt.figure(figsize=(14, 7))
    plt.plot(series.index[:train_size], series[:train_size], label='Train', color='black')
    plt.plot(test.index, test.values, label='Actual', color='blue', alpha=0.7)

    # כל חיזוי בצבע ייחודי
    for i, (pred, actual, date) in enumerate(zip(predictions, test.values, test.index)):
        color = colors[i % len(colors)]
        plt.scatter(date, pred, color=color, label=f'Pred {i+1}' if i < len(colors) else None)
        plt.plot([date, date], [pred, actual], color=color, linestyle='--', alpha=0.5)

    plt.title("Rolling Forecast - Auto ARIMA (One-Day Ahead Each Step)")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# הרצה
rolling_auto_arima(0.9)