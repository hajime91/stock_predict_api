import sys
import os
# `model`ディレクトリがある親ディレクトリをPythonのパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
#from model.lstm_model import LSTMRegressor  # モデルのインポート
#from model.cnn_model import CNNRegressor  # モデルのインポート
#from model.cnn_lstm_model import CNN_LSTM_Regressor  # モデルのインポート
#from model.timesnet_model import TimesNetRegressor  # モデルのインポート
#from model.transformer_model import TransformerRegressor  # モデルのインポート
from model.model import LSTMRegressor, CNNRegressor, CNN_LSTM_Regressor, TimesNetRegressor, TransformerRegressor  # モデルのインポート
#from data.df_processing import get_x_preds  # x_preds をインポート
#from data.df_processing import get_input_dims  # input_dims をインポート
#from data.df_processing import get_t_scalers # t_scalers をインポート
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
#from data.df_stock_data import df_stock_data_n225, df_stock_data_dow, df_stock_data_usd_jpy, df_stock_data_sp500, df_stock_data_nasdaq
from datetime import datetime, timedelta
import yfinance as yf
#import talib as ta
import mplfinance as mpf
import seaborn as sns
sns.set()
import japanize_matplotlib

# モジュール探索パスの表示
#print("Current sys.path:")
#for path in sys.path:
    #print(path)

# `model`ディレクトリの存在を確認
#model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
#print(f"Model directory exists: {os.path.exists(model_path)}")

#sys.path.append('C:\\Users\\tsuka\\OneDrive\\デスクトップ\\stock_predict_api_2\\venv\\Lib\\site-packages')





# FastAPIアプリケーションのインスタンスを作成
app = FastAPI()

# トップページ
@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Prediction API!"}

#---------------------------------------------------------------------------------------------------------------------#

# チャート
#from data.defs.def_company_stock_technical import company_stock_technical # チャート表示関数

st.title('株価チャート') # タイトル
st.write('銘柄、期間の開始日と終了日を選択してください')
tickers = {
    '^N225':"日経平均株価",
    '^DJI':"NYダウ",
    '^GSPC':"S&P500",
    '^IXIC':"NASDAQ"
    }

today = datetime.today() # 今日の日付を取得
yesterday = today - timedelta(days=1) # 前日の日付を取得
ticker = st.selectbox("銘柄", list(tickers.keys()), format_func=lambda x: tickers[x]) # ティッカーシンボルの入力
start_date = st.date_input('開始日', yesterday - timedelta(days=365)) # 日付範囲の入力
end_date = st.date_input('終了日', yesterday) # 日付範囲の入力

# 初期化 (最初に `show_graph` が存在しない場合に False を設定)
if 'show_graph1' not in st.session_state:
    st.session_state.show_graph1 = False
    
if st.button('株価チャートを表示'): # ティッカーシンボルが入力された場合、チャートを表示
    st.session_state.show_graph1 = True

# グラフを表示するかどうかを確認
if st.session_state.show_graph1:
    
    # チャートを作成する関数
    def company_stock_technical(ticker, start=None, end=None):
        
        # start や end が指定されていない場合のデフォルト値
        if start is None or end is None:
            # 今日の日付を取得
            today = datetime.today()

            # 前日の日付を取得
            yesterday = today - timedelta(days=1)
            # 1年前の日付を取得
            one_year_ago = yesterday - timedelta(days=365)

            # 日付をフォーマット (年-月-日) に変更
            if start is None:
                start = one_year_ago.strftime('%Y-%m-%d')
            if end is None:
                end = yesterday.strftime('%Y-%m-%d')

        # 株価データを取得(期間指定)
        df = yf.download(ticker, start=start, end=end)

        # タイムゾーンを削除
        df.index = df.index.tz_localize(None)

        # dfが2段のカラムを持つDataFrameの場合
        df.columns = df.columns.get_level_values(0)

        # Yahoo Financeからティッカー情報を取得
        stock_info = yf.Ticker(ticker)
        #company_name = stock_info.info.get('longName', 'Company name not found')
        # ローソク足データ
        df_candle = pd.DataFrame(df[['Open', 'High', 'Low', 'Close', 'Volume']])

        # MACD
        # 0以上:上昇トレンド
        # 0以下:下降トレンド
        # MACD_hist = MACD - MACD_signal
        #df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # RSI
        #df['RSI'] = ta.RSI(df['Adj Close'], timeperiod=14)

        # ボリンジャーバンド
        # 2σに入る
        # matype=0:単純移動平均
        # matype=1:指数移動平均
        # matype=2:加重移動平均
        #df['upper'], df['middle'], df['lower'] = ta.BBANDS(df['Adj Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        # date, price
        date = df.index
        price = df['Adj Close']

        # 単純移動平均
        df['SMA05'] = df['Adj Close'].rolling(window=5).mean()
        df['SMA25'] = df['Adj Close'].rolling(window=25).mean()
        df['SMA75'] = df['Adj Close'].rolling(window=75).mean()
        df['SMA200'] = df['Adj Close'].rolling(window=200).mean()

        # 基準線:過去26日間の最高値と最安値の平均を結んだ線
        max26 = df['High'].rolling(window=26).max()
        min26 = df['Low'].rolling(window=26).min()
        df['base_line'] = (max26 + min26) / 2

        # 転換線:過去9日間の最高値と最安値の平均を結んだ線
        high9 = df['High'].rolling(window=9).max()
        low9 = df['Low'].rolling(window=9).min()
        df['turn_line'] = (high9 + low9) / 2

        # 先行スパン1:基準線と転換線の平均を結んだ線を26日分未来にずらした線
        df['span1'] = (df['base_line'] + df['turn_line']) / 2
        df['span1'] = df['span1'].shift(25)

        # 先行スパン2:過去52日間の最高値と最安値の平均を結んだ線を26日分未来にずらした線
        high52 = df['High'].rolling(window=52).max()
        low52 = df['Low'].rolling(window=52).min()
        df['span2'] = (high52 + low52) / 2
        df['span2'] = df['span2'].shift(25)

        # 遅行線:当日の終値を26日分過去にずらした線、売り買いのタイミングを判断。遅行線より上なら売り、下なら買い
        df['slow_line'] = df['Adj Close'].shift(-25)

        # 配色サイト
        # https://colorhunt.co/palette/184189

        # ローソク足
        add_plots = [
        mpf.make_addplot(df['SMA05'], color='blue', label='SMA 05'),
        mpf.make_addplot(df['SMA25'], color='orange', label='SMA 25'),
        mpf.make_addplot(df['SMA75'], color='green', label='SMA 75'),
        # mpf.make_addplot(df['SMA200'], color='red', label='SMA 200'),
        #mpf.make_addplot(df['upper'], color='#D2E0FB'), # label='Bollinger Upper'
        # mpf.make_addplot(df['middle'], color='gray', label='Bollinger Middle'),
        #mpf.make_addplot(df['lower'], color='#D2E0FB'), # label='Bollinger Lower'
        #mpf.make_addplot(df['MACD_hist'], type='bar', color='#16423C', width=1.0, panel=3, alpha=0.5, ylabel='MACD'),
        #mpf.make_addplot(df['RSI'], type='line', color='#FF885B', panel=2, ylabel='RSI'),
        mpf.make_addplot(df_candle, type='candle', panel=1),
        mpf.make_addplot(df['base_line'], panel=1, label='Base line'), # 基準線
        mpf.make_addplot(df['turn_line'], panel=1, label='Turn line'), # 転換線
        mpf.make_addplot(df['slow_line'], panel=1, label='Slow line'), # 遅行線
        mpf.make_addplot(df[['span1', 'span2']], fill_between=dict(y1=df['span1'].values, y2=df['span2'].values, color='#CDC2A5'), panel=1, color='#CDC2A5')
        ]
        fig, axlist = mpf.plot(df_candle, addplot=add_plots, type='candle', xrotation=0, volume=True, volume_panel=2, panel_ratios=(5, 5, 1), style='yahoo',figsize=(30,30), returnfig=True) # panel_ratios:グラフの大きさの比率
        #fig.suptitle(company_name, fontsize=20) # グラフタイトルの追加

        # グラフを表示
        return fig
    
    if ticker:  # ティッカーシンボルが入力されていれば実行
        fig = company_stock_technical(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        st.pyplot(fig)  # Matplotlibの図を表示
    else:
        st.write("ティッカーシンボルを入力してください")

# 条件変更時にリセット
#if ticker or start_date or end_date:
    #st.session_state.show_graph1 = False

#---------------------------------------------------------------------------------------------------------------------#

# 株価予測

# 分析データ作成の関数
import yfinance as yf
#import talib as ta
import numpy as np

st.title('株価予測')
st.write('銘柄、学習モデルを選択してください')

# モデルタイプ、データタイプの選択肢
data_types = st.selectbox(
    '銘柄を選択してください:',
    ['日経平均株価', 'NYダウ', 'S&P500', 'NASDAQ']
)
if data_types == '日経平均株価':
    data_type = 'nikkei'
elif data_types == 'NYダウ':
    data_type = 'dow'
elif data_types == 'S&P500':
    data_type = 'sp500'
elif data_types == 'NASDAQ':
    data_type = 'nasdaq'


model_types = st.selectbox(
    '学習モデルを選択してください:',
    ['LSTM', 'CNN', 'CNN-LSTM', 'TimesNet'] # 'Transformer'
)

# 今日の日付を取得
today = datetime.today()

# 日付をフォーマット (年-月-日) に変更
start = '1800-01-01'
end = today.strftime('%Y-%m-%d')

# ウィンドウサイズ
window = 30

# 最適化されたパラメータの読み込み
# nikkei
with open('best_params_lstm_nikkei.json', 'r') as f:
    best_params_lstm_nikkei = json.load(f)
with open('best_params_cnn_nikkei.json', 'r') as f:
    best_params_cnn_nikkei = json.load(f)
with open('best_params_cnn_lstm_nikkei.json', 'r') as f:
    best_params_cnn_lstm_nikkei = json.load(f)
with open('best_params_timesnet_nikkei.json', 'r') as f:
    best_params_timesnet_nikkei = json.load(f)
with open('best_params_transformer_nikkei.json', 'r') as f:
    best_params_transformer_nikkei = json.load(f)

# dow
with open('best_params_lstm_dow.json', 'r') as f:
    best_params_lstm_dow = json.load(f)
with open('best_params_cnn_dow.json', 'r') as f:
    best_params_cnn_dow = json.load(f)
with open('best_params_cnn_lstm_dow.json', 'r') as f:
    best_params_cnn_lstm_dow = json.load(f)
with open('best_params_timesnet_dow.json', 'r') as f:
    best_params_timesnet_dow = json.load(f)
with open('best_params_transformer_dow.json', 'r') as f:
    best_params_transformer_dow = json.load(f)

# sp500
with open('best_params_lstm_sp500.json', 'r') as f:
    best_params_lstm_sp500 = json.load(f)
with open('best_params_cnn_sp500.json', 'r') as f:
    best_params_cnn_sp500 = json.load(f)
with open('best_params_cnn_lstm_sp500.json', 'r') as f:
    best_params_cnn_lstm_sp500 = json.load(f)
with open('best_params_timesnet_sp500.json', 'r') as f:
    best_params_timesnet_sp500 = json.load(f)
with open('best_params_transformer_sp500.json', 'r') as f:
    best_params_transformer_sp500 = json.load(f)

# nasdaq
with open('best_params_lstm_nasdaq.json', 'r') as f:
    best_params_lstm_nasdaq = json.load(f)
with open('best_params_cnn_nasdaq.json', 'r') as f:
    best_params_cnn_nasdaq = json.load(f)
with open('best_params_cnn_lstm_nasdaq.json', 'r') as f:
    best_params_cnn_lstm_nasdaq = json.load(f)
with open('best_params_timesnet_nasdaq.json', 'r') as f:
    best_params_timesnet_nasdaq = json.load(f)
with open('best_params_transformer_nasdaq.json', 'r') as f:
    best_params_transformer_nasdaq = json.load(f)

# 初期化 (最初に `show_graph` が存在しない場合に False を設定)
if 'show_graph2' not in st.session_state:
    st.session_state.show_graph2 = False
    
# 予測ボタン
if st.button('予測を実行'):
    st.session_state.show_graph2 = True  # グラフを表示する状態に設定
    
if st.session_state.show_graph2: # グラフを表示するかどうかを確認

    def company_stock_data(ticker, start, end):

        # 株価データを取得(期間指定)
        df = yf.download(ticker, start=start, end=end)

        # タイムゾーンを削除
        df.index = df.index.tz_localize(None)

        # dfが2段のカラムを持つDataFrameの場合
        df.columns = df.columns.get_level_values(0)

        # Yahoo Financeからティッカー情報を取得
        stock_info = yf.Ticker(ticker)
        company_name = stock_info.info.get('longName', 'Company name not found')

        # MACD
        # 0以上:上昇トレンド
        # 0以下:下降トレンド
        # MACD_hist = MACD - MACD_signal
        #df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # RSI
        #df['RSI'] = ta.RSI(df['Adj Close'], timeperiod=14)

        # ボリンジャーバンド
        # 2σに入る
        # matype=0:単純移動平均
        # matype=1:指数移動平均
        # matype=2:加重移動平均
        #df['upper'], df['middle'], df['lower'] = ta.BBANDS(df['Adj Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        # date, price
        date = df.index
        price = df['Adj Close']

        # 単純移動平均
        df['SMA05'] = df['Adj Close'].rolling(window=5).mean()
        df['SMA25'] = df['Adj Close'].rolling(window=25).mean()
        df['SMA75'] = df['Adj Close'].rolling(window=75).mean()
        df['SMA200'] = df['Adj Close'].rolling(window=200).mean()

        # 基準線:過去26日間の最高値と最安値の平均を結んだ線
        max26 = df['High'].rolling(window=26).max()
        min26 = df['Low'].rolling(window=26).min()
        df['base_line'] = (max26 + min26) / 2

        # 転換線:過去9日間の最高値と最安値の平均を結んだ線
        high9 = df['High'].rolling(window=9).max()
        low9 = df['Low'].rolling(window=9).min()
        df['turn_line'] = (high9 + low9) / 2

        # 先行スパン1:基準線と転換線の平均を結んだ線を26日分未来にずらした線
        df['span1'] = (df['base_line'] + df['turn_line']) / 2
        df['span1'] = df['span1'].shift(25)

        # 先行スパン2:過去52日間の最高値と最安値の平均を結んだ線を26日分未来にずらした線
        high52 = df['High'].rolling(window=52).max()
        low52 = df['Low'].rolling(window=52).min()
        df['span2'] = (high52 + low52) / 2
        df['span2'] = df['span2'].shift(25)

        # 遅行線:当日の終値を26日分過去にずらした線、売り買いのタイミングを判断。遅行線より上なら売り、下なら買い
        df['slow_line'] = df['Adj Close'].shift(-25)

        # 単純移動平均
        # High
        df['High_SMA05'] = df['High'].rolling(window=5).mean()
        df['High_SMA25'] = df['High'].rolling(window=25).mean()
        df['High_SMA75'] = df['High'].rolling(window=75).mean()
        df['High_SMA200'] = df['High'].rolling(window=200).mean()

        # Low
        df['Low_SMA05'] = df['Low'].rolling(window=5).mean()
        df['Low_SMA25'] = df['Low'].rolling(window=25).mean()
        df['Low_SMA75'] = df['Low'].rolling(window=75).mean()
        df['Low_SMA200'] = df['Low'].rolling(window=200).mean()

        # Open
        df['Open_SMA05'] = df['Open'].rolling(window=5).mean()
        df['Open_SMA25'] = df['Open'].rolling(window=25).mean()
        df['Open_SMA75'] = df['Open'].rolling(window=75).mean()
        df['Open_SMA200'] = df['Open'].rolling(window=200).mean()

        # 加重移動平均
        # 重みのリスト
        weights = np.array([1, 2, 3])  # 直近日間に重みをつける

        # WMAを計算
        def weighted_moving_average(x):
            return np.dot(x, weights) / weights.sum()

        # 加重移動平均を適用
        df['Adj Close WMA'] = df['Adj Close'].rolling(window=3).apply(weighted_moving_average, raw=True)
        df['Open WMA'] = df['Open'].rolling(window=3).apply(weighted_moving_average, raw=True)
        df['High WMA'] = df['High'].rolling(window=3).apply(weighted_moving_average, raw=True)
        df['Low WMA'] = df['Low'].rolling(window=3).apply(weighted_moving_average, raw=True)

        # 指数平滑移動平均
        # spanが小さいほど直近の値に重みを多く与える
        df['Adj Close EMA'] = df['Adj Close'].ewm(span=3, adjust=False).mean()
        df['Open EMA'] = df['Open'].ewm(span=3, adjust=False).mean()
        df['High EMA'] = df['High'].ewm(span=3, adjust=False).mean()
        df['Low EMA'] = df['Low'].ewm(span=3, adjust=False).mean()

        # dfを返す
        return df

    # 株価予測のデータを作成
    #from data.defs.def_company_stock_data import company_stock_data # 関数のインポート
    #import datetime
    #import pandas as pd
    from data.defs.def_processing_model import processing_lstm, processing_cnn, processing_cnn_lstm, processing_timesnet, processing_transformer

    if data_types == '日経平均株価':
        
        # 日経225株価データを取得
        ticker = '^N225'
        df_stock_data_n225 = pd.DataFrame()
        df_stock_data_n225 = company_stock_data(ticker, start, end)
        from data.df_columns import df_columns # カラム
        df_stock_data_n225 = df_stock_data_n225[df_stock_data_n225.columns.intersection(df_columns)]
        nikkei_x_pred_lstm, nikkei_input_dim_lstm, nikkei_t_scaler_lstm = processing_lstm(df_stock_data_n225, window)
        nikkei_x_pred_cnn, nikkei_input_dim_cnn, nikkei_t_scaler_cnn = processing_cnn(df_stock_data_n225, window)
        nikkei_x_pred_cnn_lstm, nikkei_input_dim_cnn_lstm, nikkei_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_n225, window)
        nikkei_x_pred_timesnet, nikkei_input_dim_timesnet, nikkei_t_scaler_timesnet = processing_timesnet(df_stock_data_n225, window)
        nikkei_x_pred_transformer, nikkei_input_dim_transformer, nikkei_t_scaler_transformer = processing_transformer(df_stock_data_n225, window)
        print(df_stock_data_n225.tail(1)) # df_stock_data_n225 の確認

    elif data_types == 'NYダウ':
        # ダウデータを取得
        ticker = '^DJI'
        df_stock_data_dow = pd.DataFrame()
        df_stock_data_dow = company_stock_data(ticker, start, end)
        from data.df_columns import df_columns # カラム
        df_stock_data_dow = df_stock_data_dow[df_stock_data_dow.columns.intersection(df_columns)]
        dow_x_pred_lstm, dow_input_dim_lstm, dow_t_scaler_lstm = processing_lstm(df_stock_data_dow, window)
        dow_x_pred_cnn, dow_input_dim_cnn, dow_t_scaler_cnn = processing_cnn(df_stock_data_dow, window)
        dow_x_pred_cnn_lstm, dow_input_dim_cnn_lstm, dow_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_dow, window)
        dow_x_pred_timesnet, dow_input_dim_timesnet, dow_t_scaler_timesnet = processing_timesnet(df_stock_data_dow, window)
        dow_x_pred_transformer, dow_input_dim_transformer, dow_t_scaler_transformer = processing_transformer(df_stock_data_dow, window)
        print(df_stock_data_dow.tail(1)) # df_stock_data_dow の確認

    elif data_types == 'S&P500':
        # SP500 のデータを取得
        ticker = '^GSPC'
        df_stock_data_sp500 = pd.DataFrame()
        df_stock_data_sp500 = company_stock_data(ticker, start, end)
        from data.df_columns import df_columns # カラム
        df_stock_data_sp500 = df_stock_data_sp500[df_stock_data_sp500.columns.intersection(df_columns)]
        sp500_x_pred_lstm, sp500_input_dim_lstm, sp500_t_scaler_lstm = processing_lstm(df_stock_data_sp500, window)
        sp500_x_pred_cnn, sp500_input_dim_cnn, sp500_t_scaler_cnn = processing_cnn(df_stock_data_sp500, window)
        sp500_x_pred_cnn_lstm, sp500_input_dim_cnn_lstm, sp500_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_sp500, window)
        sp500_x_pred_timesnet, sp500_input_dim_timesnet, sp500_t_scaler_timesnet = processing_timesnet(df_stock_data_sp500, window)
        sp500_x_pred_transformer, sp500_input_dim_transformer, sp500_t_scaler_transformer = processing_transformer(df_stock_data_sp500, window)
        print(df_stock_data_sp500.tail(1)) # df_stock_data_sp500 の確認

    elif data_types == 'NASDAQ':
        # NASDAQ のデータを取得
        ticker = '^IXIC'
        df_stock_data_nasdaq = pd.DataFrame()
        df_stock_data_nasdaq = company_stock_data(ticker, start, end)
        from data.df_columns import df_columns # カラム
        df_stock_data_nasdaq = df_stock_data_nasdaq[df_stock_data_nasdaq.columns.intersection(df_columns)]
        nasdaq_x_pred_lstm, nasdaq_input_dim_lstm, nasdaq_t_scaler_lstm = processing_lstm(df_stock_data_nasdaq, window)
        nasdaq_x_pred_cnn, nasdaq_input_dim_cnn, nasdaq_t_scaler_cnn = processing_cnn(df_stock_data_nasdaq, window)
        nasdaq_x_pred_cnn_lstm, nasdaq_input_dim_cnn_lstm, nasdaq_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_nasdaq, window)
        nasdaq_x_pred_timesnet, nasdaq_input_dim_timesnet, nasdaq_t_scaler_timesnet = processing_timesnet(df_stock_data_nasdaq, window)
        nasdaq_x_pred_transformer, nasdaq_input_dim_transformer, nasdaq_t_scaler_transformer = processing_transformer(df_stock_data_nasdaq, window)
        print(df_stock_data_nasdaq.tail(1)) # df_stock_data_nasdaq の確認


    # シーケンスデータの作成
    #from data.df_stock_data import df_stock_data_n225, df_stock_data_dow, df_stock_data_sp500, df_stock_data_nasdaq
    

    # 各データセットとモデルごとにx_predをまとめる関数
    def get_x_preds():
        if data_types == '日経平均株価':
            return {
                "nikkei": {
                    "LSTM": nikkei_x_pred_lstm,
                    "CNN": nikkei_x_pred_cnn,
                    "CNN-LSTM": nikkei_x_pred_cnn_lstm,
                    "TimesNet": nikkei_x_pred_timesnet,
                    "Transformer": nikkei_x_pred_transformer
                }
            }
            
        elif data_types == 'NYダウ':
            return {
                "dow": {
                    "LSTM": dow_x_pred_lstm,
                    "CNN": dow_x_pred_cnn,
                    "CNN-LSTM": dow_x_pred_cnn_lstm,
                    "TimesNet": dow_x_pred_timesnet,
                    "Transformer": dow_x_pred_transformer
                }
            }
            
        elif data_types == 'S&P500':
            return {
                "sp500": {
                    "LSTM": sp500_x_pred_lstm,
                    "CNN": sp500_x_pred_cnn,
                    "CNN-LSTM": sp500_x_pred_cnn_lstm,
                    "TimesNet": sp500_x_pred_timesnet,
                    "Transformer": sp500_x_pred_transformer
                }
            }
        
        elif data_types == 'NASDAQ':
            return {
                "nasdaq": {
                    "LSTM": nasdaq_x_pred_lstm,
                    "CNN": nasdaq_x_pred_cnn,
                    "CNN-LSTM": nasdaq_x_pred_cnn_lstm,
                    "TimesNet": nasdaq_x_pred_timesnet,
                    "Transformer": nasdaq_x_pred_transformer
                }
            }

    # 各データセットとモデルごとにinput_dimをまとめる関数
    def get_input_dims():
        if data_types == '日経平均株価':
            return {
                "nikkei": {
                    "LSTM": nikkei_input_dim_lstm,
                    "CNN": nikkei_input_dim_cnn,
                    "CNN-LSTM": nikkei_input_dim_cnn_lstm,
                    "TimesNet": nikkei_input_dim_timesnet,
                    "Transformer": nikkei_input_dim_transformer
                }
            }
        
        elif data_types == 'NYダウ':
            return {
                "dow": {
                    "LSTM": dow_input_dim_lstm,
                    "CNN": dow_input_dim_cnn,
                    "CNN-LSTM": dow_input_dim_cnn_lstm,
                    "TimesNet": dow_input_dim_timesnet,
                    "Transformer": dow_input_dim_transformer
                }
            }
        
        elif data_types == 'S&P500':
            return {
                "sp500": {
                    "LSTM": sp500_input_dim_lstm,
                    "CNN": sp500_input_dim_cnn,
                    "CNN-LSTM": sp500_input_dim_cnn_lstm,
                    "TimesNet": sp500_input_dim_timesnet,
                    "Transformer": sp500_input_dim_transformer
                }
            }
        
        elif data_types == 'NASDAQ':
            return {
                "nasdaq": {
                    "LSTM": nasdaq_input_dim_lstm,
                    "CNN": nasdaq_input_dim_cnn,
                    "CNN-LSTM": nasdaq_input_dim_cnn_lstm,
                    "TimesNet": nasdaq_input_dim_timesnet,
                    "Transformer": nasdaq_input_dim_transformer
                }
            }

    # 各データセットとモデルごとにt_scalerをまとめる関数
    def get_t_scalers():
        if data_types == '日経平均株価':
            return {
                "nikkei": {
                    "LSTM": nikkei_t_scaler_lstm,
                    "CNN": nikkei_t_scaler_cnn,
                    "CNN-LSTM": nikkei_t_scaler_cnn_lstm,
                    "TimesNet": nikkei_t_scaler_timesnet,
                    "Transformer": nikkei_t_scaler_transformer
                }
            }
        
        elif data_types == 'NYダウ':
            return {
                "dow": {
                "LSTM": dow_t_scaler_lstm,
                "CNN": dow_t_scaler_cnn,
                "CNN-LSTM": dow_t_scaler_cnn_lstm,
                "TimesNet": dow_t_scaler_timesnet,
                "Transformer": dow_t_scaler_transformer
                }
            }
        
        elif data_types == 'S&P500':
            return {
                "sp500": {
                    "LSTM": sp500_t_scaler_lstm,
                    "CNN": sp500_t_scaler_cnn,
                    "CNN-LSTM": sp500_t_scaler_cnn_lstm,
                    "TimesNet": sp500_t_scaler_timesnet,
                    "Transformer": sp500_t_scaler_transformer
                }
            }
        
        elif data_types == 'NASDAQ':
            return {
                "nasdaq": {
                    "LSTM": nasdaq_t_scaler_lstm,
                    "CNN": nasdaq_t_scaler_cnn,
                    "CNN-LSTM": nasdaq_t_scaler_cnn_lstm,
                    "TimesNet": nasdaq_t_scaler_timesnet,
                    "Transformer": nasdaq_t_scaler_transformer
                }
            }

    # 関数を呼び出して辞書を取得
    x_preds = get_x_preds()
    input_dims = get_input_dims()
    t_scalers = get_t_scalers()

    # 入力データの形式を定義（JSONリクエスト用）
    class PredictionRequest(BaseModel):
        model_type: str  # 'LSTM', 'CNN', 'CNN-LSTM', 'TimesNet', 'Transformer'
        data_type: str   # 'nikkei', 'dow', 'usd_jpy', 'wti', 'gold', 'us_10yb', 'nikkei_cme'
        features: list

    # モデルのロード
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデル選択の関数
    def load_model(model_type, data_type):
        # data_type と model_type に基づいて input_dim と最適パラメータを取得
        try:
            input_dim = input_dims[data_type][model_type]
        except KeyError:
            raise ValueError(f"Invalid data_type '{data_type}' or model_type '{model_type}'. Please check the available options.")
        
        # データタイプに基づいて適切なパラメータを選択
        if data_type == 'nikkei':
            if model_type == 'LSTM':
                best_params = best_params_lstm_nikkei
            elif model_type == 'CNN':
                best_params = best_params_cnn_nikkei
            elif model_type == 'CNN-LSTM':
                best_params = best_params_cnn_lstm_nikkei
            elif model_type == 'TimesNet':
                best_params = best_params_timesnet_nikkei
            elif model_type == 'Transformer':
                best_params = best_params_transformer_nikkei
            else:
                raise ValueError("Invalid model_type. Choose either 'LSTM' or 'CNN'.")
            
        elif data_type == 'dow':
            if model_type == 'LSTM':
                best_params = best_params_lstm_dow
            elif model_type == 'CNN':
                best_params = best_params_cnn_dow
            elif model_type == 'CNN-LSTM':
                best_params = best_params_cnn_lstm_dow
            elif model_type == 'TimesNet':
                best_params = best_params_timesnet_dow
            elif model_type == 'Transformer':
                best_params = best_params_transformer_dow
            else:
                raise ValueError("Invalid model_type. Choose either 'LSTM' or 'CNN'.")
            
        elif data_type == 'sp500':
            if model_type == 'LSTM':
                best_params = best_params_lstm_sp500
            elif model_type == 'CNN':
                best_params = best_params_cnn_sp500
            elif model_type == 'CNN-LSTM':
                best_params = best_params_cnn_lstm_sp500
            elif model_type == 'TimesNet':
                best_params = best_params_timesnet_sp500
            elif model_type == 'Transformer':
                best_params = best_params_transformer_sp500
            else:
                raise ValueError("Invalid model_type. Choose either 'LSTM' or 'CNN'.")
            
        elif data_type == 'nasdaq':
            if model_type == 'LSTM':
                best_params = best_params_lstm_nasdaq
            elif model_type == 'CNN':
                best_params = best_params_cnn_nasdaq
            elif model_type == 'CNN-LSTM':
                best_params = best_params_cnn_lstm_nasdaq
            elif model_type == 'TimesNet':
                best_params = best_params_timesnet_nasdaq
            elif model_type == 'Transformer':
                best_params = best_params_transformer_nasdaq
            else:
                raise ValueError("Invalid model_type. Choose either 'LSTM' or 'CNN'.")
            
        else:
            raise ValueError(f"Invalid data_type '{data_type}'. Choose either 'nikkei' or 'dow'.")
        
        # モデルの作成とロード
        if model_type == 'LSTM':
            model = LSTMRegressor(
                input_dim=input_dim,
                hidden_size=best_params['hidden_size'],
                num_layers=best_params['num_layers'],
                dropout_prob=best_params['dropout_prob'],
                lr=best_params['lr'],
                weight_decay=best_params['weight_decay']
            ).to(device)
            
        elif model_type == 'CNN':
            model = CNNRegressor(
                input_dim=input_dim,
                num_filters=best_params['num_filters'],
                kernel_size=best_params['kernel_size'],
                dropout_prob=best_params['dropout_prob'],
                lr=best_params['lr'],
                weight_decay=best_params['weight_decay']
            ).to(device)
            
        elif model_type == 'CNN-LSTM':
            model = CNN_LSTM_Regressor(
                input_dim=input_dim,
                num_filters=best_params['num_filters'],
                kernel_size=best_params['kernel_size'],
                hidden_size=best_params['hidden_size'],
                dropout_prob=best_params['dropout_prob'],
                lr=best_params['lr'],
                weight_decay=best_params['weight_decay']
            ).to(device)
            
        elif model_type == 'TimesNet':
            model = TimesNetRegressor(
                input_dim=input_dim,
                num_layers=best_params['num_layers'],
                hidden_size=best_params['hidden_size'],
                dropout_prob=best_params['dropout_prob'],
                lr=best_params['lr'],
                weight_decay=best_params['weight_decay']
            ).to(device)
            
        elif model_type == 'Transformer':
            model = TransformerRegressor(
                input_dim=input_dim,
                num_layers=best_params['num_layers'],
                num_heads=best_params['num_heads'],
                hidden_size=best_params['hidden_size'],
                dropout_prob=best_params['dropout_prob'],
                lr=best_params['lr'],
                weight_decay=best_params['weight_decay']
            ).to(device)
        
        # モデルの重みをロード
        model.load_state_dict(torch.load(f'model/stock_price_forecast_regressor_{model_type.lower()}_{data_type}.pt'))
        model.eval()  # 推論モードに設定
        return model

    # 予測エンドポイント
    @app.post("/predict/")
    def predict(request: PredictionRequest):
        model_type = request.model_type
        data_type = request.data_type

        # 入力データにx_predを使用する
        features = x_preds[data_type][model_type].to(device)
        
        # モデルのロード
        model = load_model(model_type, data_type)

        # 予測する日数（5日間の予測）
        n_steps = 5

        # 逐次予測の結果を保存するリスト
        sequential_predictions = []

        # テストデータのコピーを作成（逐次予測でデータを更新するため）
        current_input = features.clone()

        # 逐次予測を実行
        with torch.no_grad():
            for step in range(n_steps):
                # モデルに現在の入力を渡して1日分を予測
                prediction = model(current_input)[:, 0].unsqueeze(1)

                # 予測結果を保存
                sequential_predictions.append(prediction.cpu().numpy())

                # 予測した結果を次のタイムステップの入力に組み込む
                # predictionを current_input の特徴量の次元数に合わせる
                feature_dim = current_input.size(2)  # current_inputの3次元目のサイズ（特徴量数）
                prediction = prediction.unsqueeze(2).expand(-1, -1, feature_dim)  # [1, 1, feature_dim] になる

                current_input = torch.cat([current_input[:, 1:], prediction], dim=1)

        # 逐次予測の結果を連結
        sequential_predictions = np.concatenate(sequential_predictions, axis=1)

        # 結果を元のスケールに戻す
        t_scaler = t_scalers[data_type][model_type]
        predictions_rescaled = t_scaler.inverse_transform(sequential_predictions)[0]

        # 予測結果を返す
        if data_type == 'nikkei':
            df_prediction = df_stock_data_n225[['Adj Close']].tail(1)
            df_prediction.rename(columns={'Adj Close': '終値'}, inplace=True)
            df_prediction = df_prediction.T
            df_prediction.columns = df_prediction.columns.strftime('%Y年%m月%d日')
            df_day_1_5 = pd.DataFrame(predictions_rescaled)
            df_day_1_5.columns = ['終値']
            df_day_1_5 = df_day_1_5.T
            df_day_1_5.columns = ['day_1', 'day_2', 'day_3', 'day_4', 'day_5']
            df_prediction = pd.concat([df_prediction, df_day_1_5], axis=1, join='outer')
        
        elif data_type == 'dow':
            df_prediction = df_stock_data_dow[['Adj Close']].tail(1)
            df_prediction.rename(columns={'Adj Close': '終値'}, inplace=True)
            df_prediction = df_prediction.T
            df_prediction.columns = df_prediction.columns.strftime('%Y年%m月%d日')
            df_day_1_5 = pd.DataFrame(predictions_rescaled)
            df_day_1_5.columns = ['終値']
            df_day_1_5 = df_day_1_5.T
            df_day_1_5.columns = ['day_1', 'day_2', 'day_3', 'day_4', 'day_5']
            df_prediction = pd.concat([df_prediction, df_day_1_5], axis=1, join='outer')
        
        elif data_type == 'sp500':
            df_prediction = df_stock_data_sp500[['Adj Close']].tail(1)
            df_prediction.rename(columns={'Adj Close': '終値'}, inplace=True)
            df_prediction = df_prediction.T
            df_prediction.columns = df_prediction.columns.strftime('%Y年%m月%d日')
            df_day_1_5 = pd.DataFrame(predictions_rescaled)
            df_day_1_5.columns = ['終値']
            df_day_1_5 = df_day_1_5.T
            df_day_1_5.columns = ['day_1', 'day_2', 'day_3', 'day_4', 'day_5']
            df_prediction = pd.concat([df_prediction, df_day_1_5], axis=1, join='outer')
        
        elif data_type == 'nasdaq':
            df_prediction = df_stock_data_nasdaq[['Adj Close']].tail(1)
            df_prediction.rename(columns={'Adj Close': '終値'}, inplace=True)
            df_prediction = df_prediction.T
            df_prediction.columns = df_prediction.columns.strftime('%Y年%m月%d日')
            df_day_1_5 = pd.DataFrame(predictions_rescaled)
            df_day_1_5.columns = ['終値']
            df_day_1_5 = df_day_1_5.T
            df_day_1_5.columns = ['day_1', 'day_2', 'day_3', 'day_4', 'day_5']
            df_prediction = pd.concat([df_prediction, df_day_1_5], axis=1, join='outer')
            
        return {
            "prediction": df_prediction}
                #{
                #df_prediction
                #"day_1": f"{predictions_rescaled[0]:.3f}円",
                #"day_2": f"{predictions_rescaled[1]:.3f}円",
                #"day_3": f"{predictions_rescaled[2]:.3f}円",
                #"day_4": f"{predictions_rescaled[3]:.3f}円",
                #"day_5": f"{predictions_rescaled[4]:.3f}円"
            #}
        #}


    # 入力された特徴量を適切な形式に変換
    request = PredictionRequest(
        model_type=model_types,
        data_type=data_type,
        features=x_preds[data_type][model_types].to(device)
    )

    # 予測を実行
    try:
        prediction_result = predict(request)  # 予測関数の呼び出し
        st.write('予測結果:')
        st.write(prediction_result['prediction'])
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")

# 条件変更時にリセット
#if data_types or model_types:
    #st.session_state.show_graph2 = False

#---------------------------------------------------------------------------------------------------------------------#

# ポートフォリオの最適化
st.title('ポートフォリオの最適化')
st.write('開始日、終了日を選択してください')
today = datetime.today() # 今日の日付を取得
yesterday = today - timedelta(days=1) # 前日の日付を取得
start_date2 = st.date_input('開始日', yesterday - timedelta(days=365*5), key='start_date') # 日付範囲の入力
end_date2 = st.date_input('終了日', yesterday, key='end_date') # 日付範囲の入力

# 初期化 (最初に `show_graph` が存在しない場合に False を設定)
if 'show_graph3' not in st.session_state:
    st.session_state.show_graph3 = False
    
# グラフ表示ボタン
if st.button('リターン＆リスクを表示', key='display3_graph'):
    st.session_state.show_graph3 = True  # グラフを表示する状態に設定
if st.session_state.show_graph3:

    # 日経平均株価データを取得
    ticker = '^N225'
    stock_info = yf.Ticker(ticker)
    ticker_name = stock_info.info.get('longName', 'Company name not found')
    #start=datetime.today() - timedelta(days=3650)
    #today = datetime.today() # 今日の日付を取得
    #end = today.strftime('%Y-%m-%d')

    df_nikkei = yf.download(ticker, start=start_date2.strftime('%Y-%m-%d'), end=end_date2.strftime('%Y-%m-%d')) # 株価データ
    df_nikkei.rename(columns={'Adj Close': ticker_name}, inplace=True) # df_nikkei の 'Adj Close' を ticker_name に変更
    df_nikkei = df_nikkei[ticker_name] # df_nikkei の ticker_name を抽出
    df_nikkei.index = df_nikkei.index.tz_localize(None) # タイムゾーンを削除
    # df.columns = df.columns.get_level_values(0) # dfが2段のカラムを持つDataFrameの場合

    # TOPIX-17
    topix_17_codes = pd.DataFrame(list(range(1617, 1634))) # TOPIX-17 のコードリスト
    topix_17_codes.columns = ['TOPIX_17_code'] # topix_17_codes のカラム名を 'TOPIX_17_code' に変更
    topix_17_codes['TOPIX_17_code'] = topix_17_codes['TOPIX_17_code'].astype(str) + '.T' # TOPIX_17_code に '.T' を付ける
    topix_17_codes = topix_17_codes['TOPIX_17_code'].tolist() # topix_17_codes をリスト化
    df_topix_17 = yf.download(topix_17_codes, start=start_date2.strftime('%Y-%m-%d'), end=end_date2.strftime('%Y-%m-%d')) # TOPIX_17 のデータ
    df_topix_17 = df_topix_17['Adj Close'] # df_topix_17 の 'Adj Close' を抽出
    df_topix_17.index = df_topix_17.index.tz_localize(None) # タイムゾーンを削除

    # 'TOPIX_17_code' のティッカー情報を for 文で取得
    df_ticker_name = pd.DataFrame(columns=['TOPIX_17_code', 'ticker_name']) # 空のデータフレームを作成
    for ticker in topix_17_codes:
        stock_info = yf.Ticker(ticker)# Yahoo Financeからティッカー情報を取得
        ticker_name = stock_info.info.get('longName', 'Company name not found')
        new_row = pd.DataFrame({'TOPIX_17_code': [ticker], 'ticker_name': [ticker_name]}) # 新しい行を追加するためのデータフレームを作成
        df_ticker_name = pd.concat([df_ticker_name, new_row], ignore_index=True) # pd.concat()を使って新しい行を df_ticker_name に追加
        
    df_ticker_name['ticker_name'] = df_ticker_name['ticker_name'].str.replace('NEXT FUNDS TOPIX-17 ', '') # ticker_name の'NEXT FUNDS TOPIX-17 ', 'ETF'  を削除
    df_ticker_name['ticker_name'] = df_ticker_name['ticker_name'].str.replace(' ETF', '') # ticker_name の'NEXT FUNDS TOPIX-17 ', 'ETF'  を削除
    df_topix_17.columns = df_ticker_name['ticker_name'].values # df_topix_17 のカラム名を ticker_name に変更

    # df_nikkei と df_topix_17 を結合
    df_nikkei_topix_17 = pd.concat([df_nikkei, df_topix_17], axis=1)
    df_nikkei_topix_17_log_return = np.log(df_nikkei_topix_17 / df_nikkei_topix_17.shift(1)) # 対数利益率
    df_nikkei_topix_17_return = df_nikkei_topix_17_log_return.mean() * 250 # 年率リターン
    df_nikkei_topix_17_return = pd.DataFrame(df_nikkei_topix_17_return, columns=['return']) # DataFrame
    df_nikkei_topix_17_std = df_nikkei_topix_17_log_return.std() * 250 ** 0.5 # 標準偏差
    df_nikkei_topix_17_std = pd.DataFrame(df_nikkei_topix_17_std, columns=['std']) # DataFrame

    # df_topix_17_return と df_topix_17_std を結合
    df_topix_17_return_std = pd.concat([df_nikkei_topix_17_return, df_nikkei_topix_17_std], axis=1)
    df_return_std = df_topix_17_return_std
    df_return_std['sr'] = df_return_std['return'] / df_return_std['std'] # SR


    plt.figure(figsize=(10, 10)) # df_return_std を散布図でプロット
    sns.scatterplot(data=df_return_std, x='std', y='return')
    for i, txt in enumerate(df_return_std.index): # 点にindexを表示
        plt.annotate(txt, (df_return_std.iloc[i]['std'], df_return_std.iloc[i]['return']))
    plt.xlabel('Risk')
    plt.ylabel('Return')
    #plt.title('Return & Risk') # グラフのタイトル
    st.pyplot(plt)

# 条件変更時にリセット
#if start_date2 or end_date2:
    #st.session_state.show_graph3 = False

#st.write('銘柄を選択してください（複数選択可）')

# 複数選択ボタンの追加
options = ['FOODS:食品', 'ENERGY RESOURCES:エネルギー資源', 'CONSTRUCTION & MATERIALS:建設・資材',
       'RAW MATERIALS & CHEMICALS:素材・化学', 'PHARMACEUTICAL:医薬品',
       'AUTOMOBILES TRANSPORTATION EQUIPMENT:自動車・輸送機', 'STEEL & NONFERROUS:鉄鋼・非鉄',
       'MACHINERY:機械', 'ELECTRIC & PRECISION INSTRUMENTS:電機・精密',
       'IT & SERVICES,OTHERS:情報通信・サービスその他', 'ELECTRIC POWER & GAS:電機・ガス',
       'TRANSPORTATION & LOGISTICS:運輸・物流', 'COMMERCIAL & WHOLESALE TRADE:商社・卸売',
       'RETAIL TRADE:小売', 'BANKS:銀行', 'FINANCIALS (EX BANKS):金融(除く銀行)', 'REAL ESTATE:不動産']
selected_options = st.multiselect('最適化する銘柄を選択してください（複数選択可）:', options=options)

# 選択したオプションの表示
if selected_options:
    #st.write('選択された銘柄:', ', '.join(selected_options))
    selected_options = [item.split(':')[0] for item in selected_options]
    assets = selected_options #.tolist() # 銘柄をリスト化
    selected_options_number = len(selected_options)
    df_assets = df_topix_17[assets] # df_topix_17 から assets を取得
    df_assets_ind = (df_assets / df_assets.iloc[0] * 100) # df_assets を指数化
    df_nikkei_ind = (df_nikkei / df_nikkei.iloc[0] * 100) # df_nikkei の 'Adj Close' を指数化
    df_nikkei_ind.name = 'Nikkei 225' # 'Adj Close' を 'Nikkei 225' に変更
    df_nikkei_ind.index = df_nikkei_ind.index.tz_localize(None) # タイムゾーンを削除
    df_nikkei_assets_ind = pd.concat([df_nikkei_ind, df_assets_ind], axis=1) # df_nikkei_ind と df_topix_17_ind を結合
    
    # 初期化 (最初に `show_graph` が存在しない場合に False を設定)
    if 'show_graph4' not in st.session_state:
        st.session_state.show_graph4 = False
        
    # グラフ表示ボタン
    if st.button('指数化を表示', key='display4_graph'):
        st.session_state.show_graph4 = True  # グラフを表示する状態に設定
    if st.session_state.show_graph4:
        # df_nikkei_assets_ind をプロット
        df_nikkei_assets_ind.plot(figsize=(20, 10))

        # グラフのタイトル
        #plt.title('Nikkei 225 & Portfolio')
        st.pyplot(plt)

    # 初期化 (最初に `show_graph` が存在しない場合に False を設定)
    if 'show_graph5' not in st.session_state:
        st.session_state.show_graph5 = False
        
    # 予測ボタン
    if st.button('最適化を実行', key='display5_graph'):
        st.session_state.show_graph5 = True  # グラフを表示する状態に設定
        
    if st.session_state.show_graph5: # グラフを表示するかどうかを確認
        # プログレスバーの初期化
        #progress_bar = st.progress(0)
        #progress_text = st.empty()
        
        # 配分計算
        #log_returns = np.log(df_assets / df_assets.shift(1)) # 対数利益率
        # weights_list = [] # 空のリストを用意
        # assets_columns = [] # 空のリストを用意
        # pfolio_returns = [] # 空のリストを用意
        # pfolio_volatilities = [] # 空のリストを用意
        # for x in range(10000):
        #     weights = np.random.random(selected_options_number)
        #     weights /= np.sum(weights) # ランダムなそれぞれのウェイトを全体のウェイトで割る
        #     weights_list.append(weights)
        #     assets_columns.append(assets)
        #     pfolio_returns.append(np.sum(log_returns[assets_columns[x]].dropna().mean() * weights_list[x]) * 250) # ポートフォリオの予想年率リターン
        #     pfolio_volatilities.append(np.sqrt(np.dot(weights_list[x].T, np.dot(log_returns[assets_columns[x]].cov() * 250, weights_list[x])))) # ポートフォリオの予想年率リスク

        #     # プログレスバーの更新
        #     if x % 100 == 0 or x == 9999:  # 進捗の更新頻度を調整
        #         progress = int((x + 1) / 10000 * 100)
        #         progress_bar.progress(progress)
        #         progress_text.text(f"配分計算中... {progress}% 完了")
            
        # # 配分計算完了
        # progress_bar.progress(100)
        # progress_bar.empty()

        # pfolio_returns = np.array(pfolio_returns) # ポートフォリオの期待リターン
        # pfolio_volatilities = np.array(pfolio_volatilities) # ポートフォリオのリスク
        # portfolios = pd.DataFrame({'Returns': pfolio_returns, 'Volatility': pfolio_volatilities}) # DataFrame

        # 最適化
        import optuna
        
        # プログレスバーをリセットして最適化にも適用
        progress_bar = st.progress(0)
        progress_text = st.empty()
        progress_text.text("最適化を実行中...")

        # 銘柄ごとの期待リターンとリスクの計算
        log_returns = np.log(df_assets / df_assets.shift(1)) # 対数利益率
        stocks = assets  # 選定銘柄を作成
        expected_returns = log_returns.mean() * 250  # 年率期待リターン
        risks = log_returns.std() * (250 ** 0.5)     # 年率リスク
        risk_free_rate = 0.00  # 無リスク金利 (0% と仮定)
        np.random.seed(0)

        # ポートフォリオの最適化
        def objective(trial):
            selected_indices = assets # ランダムに銘柄数のインデックスを選択
            raw_weights = np.array([trial.suggest_float(f'weight_{i}', 0.0, 1.0) for i in range(selected_options_number - 1)]) # 4つの重みを提案し、合計が1になるように正規化
            last_weight = 1.0 - np.sum(raw_weights) # 最後の重みを計算（合計が1.0になるように調整）
            if last_weight < 0: # 最後の重みが0以上でなければ無限大を返す
                return float('inf')  # 条件を満たさない場合は無限大を返す
            
            weights = np.append(raw_weights, last_weight)  # 最後の重みを追加 # 重みを正規化する
            
            if np.sum(weights) > 1.0: # 合計が1.0を超えていないか確認
                return float('inf')  # 条件を満たさない場合は無限大を返す
            
            selected_returns = expected_returns[selected_indices] # 選んだ銘柄の年率リターン
            selected_risks = risks[selected_indices] # 選んだ銘柄のリスクを計算
            cov_matrix = np.cov(selected_returns, rowvar=False) # 共分散行列を計算（年率）
            portfolio_return = np.sum(selected_returns * weights) # 年率期待リターン
            #portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)  # ポートフォリオの年率リスク
            portfolio_risk = np.sqrt(np.dot(np.dot(weights.T, cov_matrix), weights)) # ポートフォリオの年率リスク
            if portfolio_risk <= 0: # シャープレシオの計算
                return float('inf')  # リスクが0または負の値の場合は無限大のシャープレシオとして扱う

            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk

            # シャープレシオを最大化する
            return -sharpe_ratio  # シャープレシオを負にして返す

        # Optuna最適化の実行
        def optimization_progress_callback(study, trial):
            # トライアルの進捗を更新
            progress = int(len(study.trials) / 1000 * 100)
            progress_bar.progress(progress)
            progress_text.text(f"最適化中... {progress}% 完了")
        
        # Optunaで最適化
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=1000, callbacks=[optimization_progress_callback])  # トライアル数
        
        # 最適化完了
        progress_bar.progress(100)
        progress_bar.empty()
        progress_text = st.empty()

        # 最良の結果を表示
        if study.best_params:
            print("Best Sharpe Ratio:", -study.best_value)  # 負を元に戻して出力
            print("Best allocation (weights):", study.best_params)

            # 最適な銘柄インデックスを使って銘柄名を取得
            selected_stocks = [stocks[i] for i in range(selected_options_number)]  # 4つの銘柄を選択

            # 最適な重みを取得
            selected_weights = np.array([study.best_params[f'weight_{i}'] for i in range(selected_options_number - 1)])
            last_weight = 1.0 - np.sum(selected_weights)  # 最後の重みを計算
            weights = np.append(selected_weights, last_weight)  # 最後の重みを追加

            # 合計を確認
            total_weight = np.sum(weights)
            print("Total Weight:", total_weight)  # 合計の出力

            # 銘柄名と重みの表示
            for stock, weight in zip(selected_stocks, weights):
                print(f"{stock}: {weight:.4f}")
        else:
            print("No valid allocation found.")

        # 銘柄名と重みの表示
        for stock, weight in zip(selected_stocks, weights):
            print(f"{stock}: {weight:.4f}")
        
        # DataFrame に格納
        df_stock_weights = pd.DataFrame({
            'Stock': selected_stocks,
            'Weight':  [f"{w*100:.2f}%" for w in weights]
        }).reset_index(drop=True)
        
        # Streamlit で DataFrame を表示
        st.write("最適結果")
        st.dataframe(df_stock_weights.set_index('Stock'))
        
        # df_topix_17 から selected_stocks を取得
        df_portfolio = df_topix_17[selected_stocks]
        
        # ポートフォリオの期待リターン、リスク、シャープレシオ
        log_returns_selected = np.log(df_portfolio / df_portfolio.shift(1)) # 銘柄ごとの対数利益率
        expected_returns_selected = log_returns_selected.mean() * 250 # 銘柄ごとの年率期待リターン
        expected_returns_selected = expected_returns_selected * weights # 銘柄ごとの年率期待リターンに weights をかける
        expected_returns_selected = np.sum(expected_returns_selected) # ポートフォリオの期待リターン
        log_returns_selected_cov_matrix = np.cov(log_returns_selected.mean() * 250, rowvar=False) # 共分散行列を計算
        #risks_selected = np.sqrt(weights.T @ log_returns_selected_cov_matrix @ weights)  # ポートフォリオの年率リスク
        risks_selected = np.sqrt(np.dot(np.dot(weights.T, log_returns_selected_cov_matrix), weights)) # ポートフォリオの年率リスク
        #risks_selected = log_returns_selected.std() * 250 ** 0.5 # 年率リスク
        #risks_selected = (risks_selected * weights) ** 2 # weights をかけて二乗
        #risks_selected = np.sqrt(np.sum(risks_selected)) # 合計の平方根
        sharpe_ratio = (expected_returns_selected - risk_free_rate) / risks_selected # シャープレシオ

        # 結果の表示
        print('ポートフォリオの期待リターン:', expected_returns_selected)
        print('ポートフォリオの年率リスク:', risks_selected)
        print('ポートフォリオのシャープレシオ:', sharpe_ratio)
        
        
        
        
        df_portfolio_ind = (df_portfolio / df_portfolio.iloc[0] * 100)# df_portfolio を指数化
        df_portfolio_ind = df_portfolio_ind * weights # 指数に weights をかける
        df_portfolio_ind['Portfolio'] = df_portfolio_ind.sum(axis=1) # 指数の合計列を追加
        df_nikkei_portfolio_ind = pd.concat([df_portfolio_ind, df_nikkei_ind], axis=1) # df_portfolio_ind と df_nikkei_ind を結合
        
        # df_nikkei_portfolio_ind をプロット
        df_nikkei_portfolio_ind[['^N225', 'Portfolio']].plot(figsize=(20, 10))
        st.pyplot(plt)
        
        df_bm = df_return_std.loc['^N225']
        df_bm = pd.DataFrame(df_bm)
        #df_bm = df_bm.rename(columns={'^N225': 'N225'})
        df_bm = df_bm.T
        df_bm = df_bm.rename(columns={'return': 'Return', 'std':'Risk', 'sr':'Sharpe Ratio'})
        #df_bm.index = ['Nikkei 225']
        
        # リターン、リスク、SR を DataFrame に格納
        df_portfolio_performance = pd.DataFrame({
        'Return': [expected_returns_selected], 
        'Risk': [risks_selected],
        'Sharpe Ratio': [sharpe_ratio]
        })
        df_portfolio_performance.index = ['PF']
        df_evaluation = pd.concat([df_bm, df_portfolio_performance], axis=0)
        
        # Return と Risk を 100 倍して % 表示に変更
        df_evaluation['Return'] = (df_evaluation['Return'] * 100).map("{:.2f}%".format)
        df_evaluation['Risk'] = (df_evaluation['Risk'] * 100).map("{:.2f}%".format)
        st.dataframe(df_evaluation)
        #df_pf_rt_ri_sr = pd.concat([expected_returns_selected, new_row], ignore_index=True)
        
    #else:
        #st.write('銘柄が選択されていません。')

    # 条件変更時にリセット
    #if selected_options:
        #st.session_state.show_graph5 = False

else:
    st.write("銘柄が選択されていません。")