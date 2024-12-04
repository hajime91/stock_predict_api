from data.defs.def_company_stock_data import company_stock_data # 関数のインポート
#from defs.def_company_stock_data import company_stock_data # 関数のインポート
import datetime
import pandas as pd


# 株価予測のデータを作成

# 今日の日付を取得
today = datetime.datetime.today()

# 日付をフォーマット (年-月-日) に変更
start = '1800-01-01'
end = today.strftime('%Y-%m-%d')

# 日経225株価データを取得
ticker = '^N225'
df_stock_data_n225 = pd.DataFrame()
df_stock_data_n225 = company_stock_data(ticker, start, end)
from data.df_columns import df_columns # カラム
#from df_columns import df_columns # カラム
df_stock_data_n225 = df_stock_data_n225[df_stock_data_n225.columns.intersection(df_columns)]
print(df_stock_data_n225.tail(1)) # df_stock_data_n225 の確認

# 日経CME株価データを取得
ticker = 'NIY=F'
df_stock_data_nikkei_cme = pd.DataFrame()
df_stock_data_nikkei_cme = company_stock_data(ticker, start, end)
from data.df_columns import df_columns # カラム
#from df_columns import df_columns # カラム
df_stock_data_nikkei_cme = df_stock_data_nikkei_cme[df_stock_data_nikkei_cme.columns.intersection(df_columns)]
print(df_stock_data_nikkei_cme.tail(1)) # df_stock_data_nikkei_cme の確認

# ダウデータを取得
ticker = '^DJI'
df_stock_data_dow = pd.DataFrame()
df_stock_data_dow = company_stock_data(ticker, start, end)
from data.df_columns import df_columns # カラム
#from df_columns import df_columns # カラム
df_stock_data_dow = df_stock_data_dow[df_stock_data_dow.columns.intersection(df_columns)]
print(df_stock_data_dow.tail(1)) # df_stock_data_dow の確認

# ドル円のデータを取得
ticker = 'JPY=X'
df_stock_data_usd_jpy = pd.DataFrame()
df_stock_data_usd_jpy = company_stock_data(ticker, start, end)
from data.df_columns import df_columns # カラム
#from df_columns import df_columns # カラム
df_stock_data_usd_jpy = df_stock_data_usd_jpy[df_stock_data_usd_jpy.columns.intersection(df_columns)]
print(df_stock_data_usd_jpy.tail(1)) # df_stock_data_usd_jpy の確認

# GOLD のデータを取得
ticker = 'GC=F'
df_stock_data_gold = pd.DataFrame()
df_stock_data_gold = company_stock_data(ticker, start, end)
from data.df_columns import df_columns # カラム
#from df_columns import df_columns # カラム
df_stock_data_gold = df_stock_data_gold[df_stock_data_gold.columns.intersection(df_columns)]
print(df_stock_data_gold.tail(1)) # df_stock_data_gold の確認

# WTI のデータを取得
ticker = 'CL=F'
df_stock_data_wti = pd.DataFrame()
df_stock_data_wti = company_stock_data(ticker, start, end)
from data.df_columns import df_columns # カラム
#from df_columns import df_columns # カラム
df_stock_data_wti = df_stock_data_wti[df_stock_data_wti.columns.intersection(df_columns)]
print(df_stock_data_wti.tail(1)) # df_stock_data_wti の確認

# us 10yb のデータを取得
ticker = '^TNX'
df_stock_data_us_10yb = pd.DataFrame()
df_stock_data_us_10yb = company_stock_data(ticker, start, end)
from data.df_columns import df_columns # カラム
#from df_columns import df_columns # カラム
df_stock_data_us_10yb = df_stock_data_us_10yb[df_stock_data_us_10yb.columns.intersection(df_columns)]
print(df_stock_data_us_10yb.tail(1)) # df_stock_data_us_10yb の確認

# SP500 のデータを取得
ticker = '^GSPC'
df_stock_data_sp500 = pd.DataFrame()
df_stock_data_sp500 = company_stock_data(ticker, start, end)
from data.df_columns import df_columns # カラム
#from df_columns import df_columns # カラム
df_stock_data_sp500 = df_stock_data_sp500[df_stock_data_sp500.columns.intersection(df_columns)]
print(df_stock_data_sp500.tail(1)) # df_stock_data_sp500 の確認

# NASDAQ のデータを取得
ticker = '^IXIC'
df_stock_data_nasdaq = pd.DataFrame()
df_stock_data_nasdaq = company_stock_data(ticker, start, end)
from data.df_columns import df_columns # カラム
#from df_columns import df_columns # カラム
df_stock_data_nasdaq = df_stock_data_nasdaq[df_stock_data_nasdaq.columns.intersection(df_columns)]
print(df_stock_data_nasdaq.tail(1)) # df_stock_data_nasdaq の確認
