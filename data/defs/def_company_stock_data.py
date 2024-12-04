import yfinance as yf
#import talib as ta
import numpy as np

# 分析データ作成の関数
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