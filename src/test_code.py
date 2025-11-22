import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
import yfinance as yf # הספרייה החדשה לטעינת נתונים

# =========================
# 1. טעינת נתונים חיים מ-Yahoo Finance
# =========================
ticker = 'BTC-USD'
start_date = '2020-01-01'
print(f"טוען נתונים עבור {ticker} החל מ- {start_date}...")

try:
    df = yf.download(ticker, start=start_date, progress=False)
    # נשתמש רק במחיר הסגירה (Close)
    df = df[['Close']].rename(columns={'Close': 'BTC-USD'})
    df = df.dropna()
except Exception as e:
    print(f"שגיאה בטעינת הנתונים: {e}. ודא שספריית yfinance מותקנת.")
    exit()

# =========================
# 2. בדיקת טווח וסינון
# =========================
print("\n--- בדיקת טווח נתונים לאחר סינון ---")
print(f"תאריך התחלה: {df.index.min()}")
print(f"תאריך סיום: {df.index.max()}")
print(f"סה\"כ שורות: {len(df)}")
print("--------------------------------------\n")

# =========================
# 3. הגדרת יעד וטרנספורמציה (לוג)
# =========================
y = df['BTC-USD']
y_log = np.log(y)

# =========================
# 4. פיצול נתונים ל-Train/Test לפי זמן (90/10)
# =========================
train_size = int(len(df) * 0.9)
y_train_log, y_test_log = y_log.iloc[:train_size], y_log.iloc[train_size:]

# שמירת נתוני המקור (לא לוג) עבור הגרף והתיקון
y_train_orig = y.iloc[:train_size]
y_test_orig = y.iloc[train_size:]

# ==========================================================
## שלב א': חיפוש פרמטרים אוטומטי (Auto-ARIMA)
# ==========================================================

print("מתחיל חיפוש אוטומטי של פרמטרים (Auto-ARIMA) עבור SARIMA...")
auto_model = pm.auto_arima(
    y_train_log,
    start_p=0,
    start_q=0,
    max_p=3,
    max_q=3,
    m=1,               # תקופתיות (5 ימי מסחר בשבוע)
    d=None,               # אינטגרציה לא עונתית
    D=None,               # אינטגרציה עונתית
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    seasonal=True
)

best_params = auto_model.get_params()
order = best_params.get('order', (1, 1, 0)) # ברירת מחדל אם לא נמצאו
seasonal_order = best_params.get('seasonal_order', (0, 0, 0, 5))

print(f"\n--- פרמטרים מיטביים שנמצאו ---")
print(f"order (p, d, q): {order}")
print(f"seasonal_order (P, D, Q, m): {seasonal_order}")
print("--------------------------------------")

# ==========================================================
## שלב ב': תחזית מתגלגלת (Rolling Forecast)
# ==========================================================

print("\n--- מתחיל תחזית מתגלגלת (Rolling Forecast) ---")

# נתוני האימון ההתחלתיים בלוג
# Fix: Extract numerical values from the DataFrame column
history_log = y_train_log['BTC-USD'].tolist()

# רשימה לשמירת התוצאות
predictions_rolling = []

# לולאה על כל יום בסט הבדיקה (20% הנותרים)
for t in range(len(y_test_log)):

    # 1. אימון המודל (SARIMA טהור - אין exog!)
    model_roll = SARIMAX(
        history_log, # שימו לב, אין exog!
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    # Increased maxiter to help with convergence warnings
    model_fit_roll = model_roll.fit(disp=False, maxiter=1000)#, method='lbfgs'

    # 2. תחזית ליום אחד קדימה
    output = model_fit_roll.forecast()
    yhat_log = output[0]

    # 3. המרה לסקאלה המקורית
    yhat = np.exp(yhat_log)
    predictions_rolling.append(yhat)

    # 4. הכנסת נתון האמת (מה-Test) להיסטוריה
    # Fix: Extract scalar numerical value from the Series returned by iloc
    obs_log = y_test_log['BTC-USD'].iloc[t]
    history_log.append(obs_log)

    # נדפיס רק כל 20 יום כדי לא להציף
    if (t + 1) % 20 == 0 or t == len(y_test_log) - 1:
        print(f"יום {t+1}/{len(y_test_log)}: נחזה: {yhat:.2f}, אמת: {y_test_orig.iloc[t].item():.2f}")

# =========================
# 5. עיבוד תוצאות וגרף
# =========================
y_pred_rolling_series = pd.Series(predictions_rolling, index=y_test_orig.index)

mae_roll = mean_absolute_error(y_test_orig, y_pred_rolling_series)
rmse_roll = np.sqrt(mean_squared_error(y_test_orig, y_pred_rolling_series))

print("\n--- מדדי ביצועים על קבוצת ה-Test (Rolling Forecast) ---")
print(f"MAE: {mae_roll:.2f}")
print(f"RMSE: {rmse_roll:.2f}")
print("----------------------------------------------------------")

plt.figure(figsize=(14, 7))
plt.plot(y_test_orig, label='BTC-USD - נתוני אמת (Test)', color='green', alpha=0.7)
plt.plot(y_pred_rolling_series, label='BTC-USD תחזית מתגלגלת (SARIMA טהור)', color='red', linestyle='--')
plt.title(f"תחזית מתגלגלת SARIMA טהור ל-BTC-USD (מ-2020)")
plt.xlabel("תאריך")
plt.ylabel("מחיר BTC-USD")
plt.legend()
plt.grid(True)
plt.show()