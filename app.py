
import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Label'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

def train_model(df):
    features = ['Return', 'MA5', 'MA10']
    X = df[features]
    y = df['Label']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def predict_next_day(df, model):
    latest = df.iloc[-1][['Return', 'MA5', 'MA10']].values.reshape(1, -1)
    prob = model.predict_proba(latest)[0][1]
    return prob

st.set_page_config(page_title="台股預測", page_icon="💰")
st.title("台股多支股票漲跌預測")

symbols = st.text_input("輸入台股代碼（以逗號分隔，例如：2330.TW, 2303.TW）", value="2330.TW, 2303.TW")
start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"))
end_date = st.date_input("結束日期", pd.to_datetime("2024-12-31"))

if st.button("開始預測"):
    tickers = [s.strip() for s in symbols.split(",")]
    report = []

    for ticker in tickers:
        try:
            df = get_stock_data(ticker, str(start_date), str(end_date))
            df = add_features(df)
            model = train_model(df)
            prob = predict_next_day(df, model)

            st.subheader(f"{ticker} 預測結果")
            st.write(f"**明天上漲的機率為：{prob:.2%}**")

            fig, ax = plt.subplots()
            df['Close'].plot(ax=ax, title=f"{ticker} 收盤價")
            st.pyplot(fig)

            report.append({
                "股票代碼": ticker,
                "最新日期": df.index[-1].date(),
                "上漲機率": round(prob * 100, 2)
            })

        except Exception as e:
            st.error(f"{ticker} 發生錯誤：{str(e)}")

    if report:
        report_df = pd.DataFrame(report)
        st.subheader("預測報告")
        st.dataframe(report_df)
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button("下載預測報告 (CSV)", csv, file_name="prediction_report.csv", mime="text/csv")
