from data.df_stock_data import df_stock_data_n225, df_stock_data_nikkei_cme, df_stock_data_dow, df_stock_data_usd_jpx, df_stock_data_gold, df_stock_data_wti, df_stock_data_us_10yb
import pandas as pd
import numpy as np

# df に df_stock_data_n225 を格納
df = df_stock_data_n225
# df.set_index(keys='Date', inplace=True)
df = df.reset_index() # Date を index に表示
df['Date'] = pd.to_datetime(df['Date']) # dfのDateのデータ型を'datetime'型へ変更


# df_nikkei_cme に df_stock_data_nikkei_cme を格納
df_nikkei_cme = df_stock_data_nikkei_cme
# df_nikkei_cme.set_index(keys='Date', inplace=True)
df_nikkei_cme = df_nikkei_cme.reset_index() # Date を index に表示
df_nikkei_cme['Date'] = pd.to_datetime(df_nikkei_cme['Date']) # Date のデータ型を'datetime'型へ変更
df_nikkei_cme.rename(columns= {'Adj Close' : 'CME Adj Close'}, inplace=True) # Adj CloseをCME Adj Closeへ変更
df = pd.merge(df, df_nikkei_cme[['Date','CME Adj Close']], on='Date', how='left') # dfとdf_nikkei_cmeをDateでマージ
df['CME Adj Close'] = df['CME Adj Close'].ffill() # CME Adj Close の NaN を直前の値で補完
df['CME Adj Close night'] = df['CME Adj Close'].shift(+1) # CME Adj Close を1つ下げる
df['CME_EMA'] = df['CME Adj Close night'].ewm(span=3, adjust=False).mean()
df['CME_SMA05'] = df['CME Adj Close night'].rolling(window=5).mean()
df['CME_SMA25'] = df['CME Adj Close night'].rolling(window=25).mean()
df['CME_SMA75'] = df['CME Adj Close night'].rolling(window=75).mean()
df['CME_SMA200'] = df['CME Adj Close night'].rolling(window=200).mean()


# df_dow に df_stock_data_dow を格納
df_dow = df_stock_data_dow
# df_dow.set_index(keys='Date', inplace=True)
df_dow = df_dow.reset_index() # Date を index に表示
df_dow['Date'] = pd.to_datetime(df_dow['Date']) # Date のデータ型を'datetime'型へ変更
df_dow.rename(columns= {'Adj Close' : 'DOW Adj Close'}, inplace=True) # Adj CloseをCME Adj Closeへ変更
df = pd.merge(df, df_dow[['Date','DOW Adj Close']], on='Date', how='left') # dfとdf_dowをDateでマージ
df['DOW Adj Close'] = df['DOW Adj Close'].ffill() # DOW Adj Close の NaN を直前の値で補完
df['DOW Adj Close night'] = df['DOW Adj Close'].shift(+1) # DOW Adj Close を1つ下げる
df['DOW_EMA'] = df['DOW Adj Close night'].ewm(span=3, adjust=False).mean()
df['DOW_SMA05'] = df['DOW Adj Close night'].rolling(window=5).mean()
df['DOW_SMA25'] = df['DOW Adj Close night'].rolling(window=25).mean()
df['DOW_SMA75'] = df['DOW Adj Close night'].rolling(window=75).mean()
df['DOW_SMA200'] = df['DOW Adj Close night'].rolling(window=200).mean()


# df_usdjpy に df_stock_data_usdjpx を格納
df_usdjpy = df_stock_data_usdjpx
# df_usdjpy.set_index(keys='Date', inplace=True)
df_usdjpy = df_usdjpy.reset_index() # Date を index に表示
df_usdjpy['Date'] = pd.to_datetime(df_usdjpy['Date']) # Date のデータ型を'datetime'型へ変更
df_usdjpy.rename(columns= {'Adj Close' : 'USDJPY Adj Close'}, inplace=True) # Adj CloseをCME Adj Closeへ変更
df = pd.merge(df, df_usdjpy[['Date','USDJPY Adj Close']], on='Date', how='left') # dfとdf_usdjpyをDateでマージ
df['USDJPY Adj Close'] = df['USDJPY Adj Close'].ffill() # USDJPY Adj Close の NaN を直前の値で補完
df['USDJPY Adj Close night'] = df['USDJPY Adj Close'].shift(+1) # USDJPY Adj Close を1つ下げる
df['USDJPY_EMA'] = df['USDJPY Adj Close night'].ewm(span=3, adjust=False).mean()
df['USDJPY_SMA05'] = df['USDJPY Adj Close night'].rolling(window=5).mean()
df['USDJPY_SMA25'] = df['USDJPY Adj Close night'].rolling(window=25).mean()
df['USDJPY_SMA75'] = df['USDJPY Adj Close night'].rolling(window=75).mean()
df['USDJPY_SMA200'] = df['USDJPY Adj Close night'].rolling(window=200).mean()


# 'Date' を index にセット
df.set_index(keys='Date', inplace=True)


# 単純移動平均：High, Low, Open
df['High_SMA05'] = df['High'].rolling(window=5).mean()
df['High_SMA25'] = df['High'].rolling(window=25).mean()
df['High_SMA75'] = df['High'].rolling(window=75).mean()
df['High_SMA200'] = df['High'].rolling(window=200).mean()
df['Low_SMA05'] = df['Low'].rolling(window=5).mean()
df['Low_SMA25'] = df['Low'].rolling(window=25).mean()
df['Low_SMA75'] = df['Low'].rolling(window=75).mean()
df['Low_SMA200'] = df['Low'].rolling(window=200).mean()
df['Open_SMA05'] = df['Open'].rolling(window=5).mean()
df['Open_SMA25'] = df['Open'].rolling(window=25).mean()
df['Open_SMA75'] = df['Open'].rolling(window=75).mean()
df['Open_SMA200'] = df['Open'].rolling(window=200).mean()


# 加重移動平均
weights = np.array([1, 2, 3])  # 直近日間に重みをつける
def weighted_moving_average(x): # WMAを計算
    return np.dot(x, weights) / weights.sum()

df['WMA'] = df['Adj Close'].rolling(window=3).apply(weighted_moving_average, raw=True)
df['Open WMA'] = df['Open'].rolling(window=3).apply(weighted_moving_average, raw=True)
df['High WMA'] = df['High'].rolling(window=3).apply(weighted_moving_average, raw=True)
df['Low WMA'] = df['Low'].rolling(window=3).apply(weighted_moving_average, raw=True)
df['CME_WMA'] = df['CME Adj Close night'].rolling(window=3).apply(weighted_moving_average, raw=True)
df['DOW_WMA'] = df['DOW Adj Close night'].rolling(window=3).apply(weighted_moving_average, raw=True)
df['USDJPY_WMA'] = df['USDJPY Adj Close night'].rolling(window=3).apply(weighted_moving_average, raw=True)


# 指数平滑移動平均
df['EMA'] = df['Adj Close'].ewm(span=3, adjust=False).mean()
df['Open EMA'] = df['Open'].ewm(span=3, adjust=False).mean()
df['High EMA'] = df['High'].ewm(span=3, adjust=False).mean()
df['Low EMA'] = df['Low'].ewm(span=3, adjust=False).mean()


# df のカラム
from data.df_columns import df_columns
df = df[df.columns.intersection(df_columns)]


# df の確認
#print(df.head(1))
print(df.tail(1))

















