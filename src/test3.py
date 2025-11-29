import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#  הורדת נתונים של מניית אפל (AAPL) עם נתונים שבועיים
ticker = "AAPL"
df = yf.download(ticker, start="2020-01-01", end="2025-04-02")
df.to_csv(f"{ticker}.csv")
df = df[['Close']]  # שומרים רק את המחיר המתואם
df.rename(columns={'Close': 'price'}, inplace=True)

#  התאמת המודל SARIMA
# הגדרת טווחים עבור AR, MA, Seasonal AR ו- Seasonal MA
p_range = range(0, 5)
d_range = range(0, 5)
q_range = range(0, 5)
P_range = range(0, 5)
D_range = range(0, 5)
Q_range = range(0, 5)
s = 52  # עונתיות שבועית

# פונקציה להרצת SARIMA והחזרת AIC
def run_sarima(df, p, d, q, P, D, Q, s):
    try:
        model = sm.tsa.statespace.SARIMAX(df['price'], order=(p, d, q), seasonal_order=(P, D, Q, s))
        results = model.fit(disp=False)
        return results.aic
    except:
        return np.inf  # אם יש שגיאה, נחזיר אינסוף כדי להתעלם מהמודל הזה

# חיפוש אחר השילוב הטוב ביותר של פרמטרים על פי AIC
best_aic = float('inf')
best_params = None

# רשימה לשמירת התוצאות
results_list = []

# לולאות לבדוק את כל השילובים האפשריים
for p in p_range:
    for d in d_range:
        for q in q_range:
            for P in P_range:
                for D in D_range:
                    for Q in Q_range:
                        aic = run_sarima(df, p, d, q, P, D, Q, s)
                        print(f"SARIMA({p},{d},{q})x({P},{D},{Q},{s}) -> AIC: {aic}")

                        # הוספת התוצאה לרשימה
                        results_list.append(((p, d, q, P, D, Q), aic))

# יצירת DataFrame עם התוצאות
results_df = pd.DataFrame(results_list, columns=['Parameters', 'AIC'])

# הצגת התוצאות
print("\nBest Parameters and AIC:")
print(results_df)

#  פרמטרים הטובים ביותר
best_params = results_df.loc[results_df['AIC'].idxmin()]['Parameters']
print(f"\nBest Parameters: {best_params}")

#  הרצת SARIMA עם הפרמטרים הטובים ביותר
p, d, q, P, D, Q = best_params
model = sm.tsa.statespace.SARIMAX(df['price'], order=(p, d, q), seasonal_order=(P, D, Q, s))
results = model.fit(disp=False)

#  חיזוי 21 שבועות קדימה
forecast_period = 21
forecast = results.predict(start=df.index[-1] + pd.Timedelta(weeks=1),
                           end=df.index[-1] + pd.Timedelta(weeks=forecast_period))

#  יצירת ציר הזמן עבור החיזוי
forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_period+1, freq='W')[1:]

#  הדפסת גרף היסטורי + חיזוי + מניפת רווח בר-סמך
plt.figure(figsize=(12,6))
plt.plot(df.index, df['price'], label="מחיר היסטורי", color='blue')
plt.plot(forecast_dates, forecast, label="תחזית 21 שבועות קדימה", color='red')

# מניפת הרווח (95% Confidence Interval) - חישוב בעזרת הטעויות בסטטיסטיקה
conf_int = results.get_forecast(steps=forecast_period).conf_int(alpha=0.05)
plt.fill_between(forecast_dates,
                 conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color='pink', alpha=0.3, label="טווח בר סמך 95%")

plt.xlabel("תאריך")
plt.ylabel("מחיר (דולר)")
plt.title(f"חיזוי מניית {ticker} - 21 שבועות קדימה עם מניפת רווח")
plt.legend()
plt.grid()
plt.show()