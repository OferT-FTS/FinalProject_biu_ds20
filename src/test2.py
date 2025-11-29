import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# --- 0. הגדרות ---
ticker = 'TA35.TA'
arima_order = (1, 1, 1)
split_ratio = 0.9

# --- 1. הורדת נתונים ---
print(f"מוריד נתונים עבור {ticker} מהשנים האחרונות...")
end_date = datetime.now()
start_date = end_date - timedelta(days=10 * 365) # 5 שנים אחורה

data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    raise ValueError(f"לא נמצאו נתונים עבור {ticker}.")

# נוודא שמדובר בסדרה נקייה ונומרית
ts_data = data['Close'].dropna().astype(float).squeeze()

print(f"סה\"כ נתונים שנמצאו: {len(ts_data)} ימי מסחר.")
print(f"תאריך התחלה: {ts_data.index[0]}, תאריך סיום: {ts_data.index[-1]}")

# --- 2. פיצול נתונים ---
split_index = int(len(ts_data) * split_ratio)
train_data = ts_data.iloc[:split_index]
test_data = ts_data.iloc[split_index:]

print(f"גודל סט האימון (Train): {len(train_data)}")
print(f"גודל סט המבחן (Test): {len(test_data)}")

# --- 3. חיזוי מתגלגל ---
history = train_data.values.tolist()
predictions = [] # זו הרשימה שתכיל את החיזויים

print(f"\nמתחיל תהליך חיזוי מתגלגל עם ARIMA{arima_order}...")

for t in range(len(test_data)):
    try:
        # ודא שההיסטוריה היא מערך נקי ללא NaNs
        history_array = np.array(history, dtype=float)
        history_array = history_array[~np.isnan(history_array)]

        model = ARIMA(history_array, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
    except Exception as e:
        # print(f"שגיאה באימון בצעד {t}: {e}") # אפשר להדליק לדיבוג
        yhat = history[-1]  # שימוש בערך האחרון הידוע במקרה כישלון

    # --- זה החלק החשוב ---
    # הוספת החיזוי *תמיד*, בין אם הצליח או נכשל
    predictions.append(yhat)

    # הוספת הערך האמיתי להיסטוריה לאימון הבא
    actual_obs = float(test_data.iloc[t])
    history.append(actual_obs)

    if (t + 1) % 10 == 0 or t == len(test_data) - 1:
        print(f"צעד {t + 1}/{len(test_data)}: חיזוי={yhat:.2f}, ערך אמיתי={actual_obs:.2f}")

print("\n תהליך החיזוי הושלם.")

# --- 4. יצירת DataFrame וקובץ (החלק המתוקן) ---
#    כאן אנו בודקים את השגיאה 'ValueError'
try:
    results_df = pd.DataFrame({
        'Date': test_data.index,
        'Actual': test_data.values,
        'Predicted': predictions  # <-- שימוש ברשימה 'predictions' שיצרנו
    })
except ValueError as e:
    print("--- !!! אופס, השגיאה 'ValueError' עדיין קיימת !!! ---")
    print(f"השגיאה: {e}")
    print(f"אורך test_data.index: {len(test_data.index)}")
    print(f"אורך test_data.values: {len(test_data.values)}")
    print(f"אורך הרשימה 'predictions': {len(predictions)}")
    print("זה לא אמור לקרות. בדוק את לוגיקת הלולאה שלך שוב.")
    raise e

# 2. *עכשיו* ש-'results_df' קיים, אפשר לשמור אותו
results_df.to_csv('TA35_ARIMA_forecast.csv', index=False, encoding='utf-8-sig')
print(" קובץ התחזית נשמר: TA35_ARIMA_forecast.csv")


# --- 5. הערכת ביצועים ---
# אין צורך ליצור 'predictions_series' מחדש,
# אפשר להשתמש בעמודות מ-results_df
rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Predicted']))
mape = mean_absolute_percentage_error(results_df['Actual'], results_df['Predicted']) * 100

print("\n--- הערכת ביצועי המודל ---")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# --- 6. גרף ---
print("מציג גרף...")
plt.figure(figsize=(14, 7))
plt.plot(results_df['Date'], results_df['Actual'], label='Actual', color='blue', linewidth=2)
plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted', color='red', linestyle='--', linewidth=2)
plt.title(f'TA-35 ARIMA{arima_order} Forecast (10% Test Data)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('TA35_forecast_plot.png')
print("הגרף נשמר כ-TA35_forecast_plot.png")
# plt.show() # 'show' לא עובד בסביבה זו, 'savefig' כן.

# --- 7. תחזית קדימה (10 ימים) ---

future_days = 10
print(f"\nמחשב תחזית {future_days} ימים קדימה...")

# התאמת המודל מחדש על כל הסדרה (אימון + בדיקה)
full_series = ts_data.values.astype(float)

model = ARIMA(full_series, order=arima_order)
model_fit = model.fit()

future_forecast = model_fit.forecast(steps=future_days)

# יצירת תאריכים עתידיים
last_date = ts_data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                             periods=future_days,
                             freq='B')  # ימי מסחר

future_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast': future_forecast
})

# שמירה לקובץ
future_df.to_csv('TA35_future_10days_forecast.csv', index=False, encoding='utf-8-sig')
print("קובץ התחזית העתידית נשמר: TA35_future_10days_forecast.csv")

# גרף תחזית קדימה
plt.figure(figsize=(14, 7))
plt.plot(ts_data.index, ts_data.values, label='Historical', linewidth=2)
plt.plot(future_df['Date'], future_df['Forecast'], label='Future Forecast (10 days)', linestyle='--')
plt.title(f'TA-35 ARIMA{arima_order} 10-Day Ahead Forecast')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('TA35_10day_future_forecast.png')
print("גרף תחזית 10 ימים נשמר בשם TA35_10day_future_forecast.png")
