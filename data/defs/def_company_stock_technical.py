import yfinance as yf
#import talib as ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import japanize_matplotlib
from datetime import datetime, timedelta
import pandas as pd


# グラフ作成の関数

def company_stock_technical(ticker, start=None, end=None):
    
    # start や end が指定されていない場合のデフォルト値
    if start is None or end is None:
        # 今日の日付を取得
        today = datetime.today()
        # 1年前の日付を取得
        one_year_ago = today - timedelta(days=365)

        # 日付をフォーマット (年-月-日) に変更
        if start is None:
            start = one_year_ago.strftime('%Y-%m-%d')
        if end is None:
            end = today.strftime('%Y-%m-%d')

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