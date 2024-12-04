import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl

# シーケンスデータを作成する関数
def processing_lstm(df, window):
    
    # df の NaN を削除
    df.dropna(inplace=True)
    
    # df を df_processing に格納
    df_processing = df
    df_x = df_processing.drop(['Adj Close'], axis=1) # 入力データ
    df_t = df_processing[['Adj Close']] # ターゲット
    df_x_columns = df_x.columns # df_x のカラムを取得
    df_x = pd.DataFrame(df_x, columns=df_x_columns) # DataFrame に変換
    df_t = pd.DataFrame(df_t, columns=['Adj Close']) # DataFrame に変換

    # df から後ろのwindowを df_pred に格納
    df_pred = df.tail(window)
    df_pred_x = df_pred.drop(['Adj Close'], axis=1) # 入力データ
    df_pred_x = pd.DataFrame(df_pred_x, columns=df_x_columns) # DataFrame に変換

    # 入力値x, 目標値t
    # ウィンドウサイズ
    window = window
    x, t = [], []

    # 連続するウィンドウを移動させながらループ
    for time in range(len(df_processing) - window):

        # 入力値x：連続する window期間 の 'Adj Close' 以外のデータを取得
        features = df_x.iloc[time:time + window].values
        x.append(features)

        # 目標値t : 'Adj Close' を取得
        target = df_t.iloc[time + window - 1].values
        t.append(target)

    # データをNumPy配列に変換
    x = np.array(x)
    t = np.array(t)

    # df_pred_x を ndarray に変換
    x_pred = np.array(df_pred_x)
    x_pred = x_pred.reshape(-1, window, x_pred.shape[1]) # x_pred を3次元に変換

    # 入力値x の特徴量数
    input_dim = x.shape[2]
    x = x.reshape(-1, window, input_dim) # 入力値xの shape を (batch_size, seq_len) -> (batch_size, seq_len, input_dim) へ変換
    t = t.reshape(-1, 1) # 目標値t shape を (batch_size) -> (batch_size, input_dim) へ変換
    x_pred = x_pred.reshape(-1, window, input_dim) # 予測入力値x_pred の shape を (batch_size, seq_len) -> (batch_size, seq_len, input_dim) へ変換

    # Tensor に変換
    x = torch.tensor(x, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    x_pred = torch.tensor(x_pred, dtype=torch.float32)

    # train, val, test のサンプルサイズを決定
    n_train = int(len(x) * 0.8)
    n_val = int(len(x) * 0.1)
    n_test = len(x) - n_train - n_val

    # train, val, test に分割
    x_train, t_train = x[0: n_train], t[0: n_train]
    x_val, t_val = x[n_train: n_train+n_val], t[n_train: n_train+n_val]
    x_test, t_test = x[n_train+n_val:], t[n_train+n_val:]
    
    # Tensorをndarrayに変換
    x_train = x_train.numpy()
    t_train = t_train.numpy()
    x_val = x_val.numpy()
    t_val = t_val.numpy()
    x_test = x_test.numpy()
    t_test = t_test.numpy()
    x_pred = x_pred.numpy()

    # 3次元から2次元に変換
    x_train = x_train.reshape(-1, window*input_dim)
    x_val = x_val.reshape(-1, window*input_dim)
    x_test = x_test.reshape(-1, window*input_dim)
    x_pred = x_pred.reshape(-1, window*input_dim)

    # MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler

    # x_train で fit
    x_scaler = MinMaxScaler()
    x_scaler.fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_val = x_scaler.transform(x_val)
    x_test = x_scaler.transform(x_test)
    x_pred = x_scaler.transform(x_pred)

    # t_train で fit
    t_scaler = MinMaxScaler()
    t_scaler.fit(t_train)
    t_train = t_scaler.transform(t_train)
    t_val = t_scaler.transform(t_val)
    t_test = t_scaler.transform(t_test)

    # 2次元から3次元に戻す
    x_train = x_train.reshape(-1, window, input_dim)
    x_val = x_val.reshape(-1, window, input_dim)
    x_test = x_test.reshape(-1, window, input_dim)
    x_pred = x_pred.reshape(-1, window, input_dim)

    # ndarray を Tensor に変換
    x_train = torch.tensor(x_train, dtype=torch.float32)
    t_train = torch.tensor(t_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    t_val = torch.tensor(t_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    t_test = torch.tensor(t_test, dtype=torch.float32)
    x_pred = torch.tensor(x_pred, dtype=torch.float32)    
    
    return x_pred, input_dim, t_scaler


# シーケンスデータを作成する関数
def processing_cnn(df, window):
    
    # df の NaN を削除
    df.dropna(inplace=True)
    
    # df を df_processing に格納
    df_processing = df
    df_x = df_processing.drop(['Adj Close'], axis=1) # 入力データ
    df_t = df_processing[['Adj Close']] # ターゲット
    df_x_columns = df_x.columns # df_x のカラムを取得
    df_x = pd.DataFrame(df_x, columns=df_x_columns) # DataFrame に変換
    df_t = pd.DataFrame(df_t, columns=['Adj Close']) # DataFrame に変換

    # df から後ろのwindowを df_pred に格納
    df_pred = df.tail(window)
    df_pred_x = df_pred.drop(['Adj Close'], axis=1) # 入力データ
    df_pred_x = pd.DataFrame(df_pred_x, columns=df_x_columns) # DataFrame に変換

    # 入力値x, 目標値t
    # ウィンドウサイズ
    window = window
    x, t = [], []

    # 連続するウィンドウを移動させながらループ
    for time in range(len(df_processing) - window):

        # 入力値x：連続する window期間 の 'Adj Close' 以外のデータを取得
        features = df_x.iloc[time:time + window].values
        x.append(features)

        # 目標値t : 'Adj Close' を取得
        target = df_t.iloc[time + window - 1].values
        t.append(target)

    # データをNumPy配列に変換
    x = np.array(x)
    t = np.array(t)

    # df_pred_x を ndarray に変換
    x_pred = np.array(df_pred_x)
    x_pred = x_pred.reshape(-1, window, x_pred.shape[1]) # x_pred を3次元に変換

    # 入力値x の特徴量数
    input_dim = x.shape[2]
    x = x.reshape(-1, window, input_dim) # 入力値xの shape を (batch_size, seq_len) -> (batch_size, seq_len, input_dim) へ変換
    t = t.reshape(-1, 1) # 目標値t shape を (batch_size) -> (batch_size, input_dim) へ変換
    x_pred = x_pred.reshape(-1, window, input_dim) # 予測入力値x_pred の shape を (batch_size, seq_len) -> (batch_size, seq_len, input_dim) へ変換

    # Tensor に変換
    x = torch.tensor(x, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    x_pred = torch.tensor(x_pred, dtype=torch.float32)

    # train, val, test のサンプルサイズを決定
    n_train = int(len(x) * 0.8)
    n_val = int(len(x) * 0.1)
    n_test = len(x) - n_train - n_val

    # train, val, test に分割
    x_train, t_train = x[0: n_train], t[0: n_train]
    x_val, t_val = x[n_train: n_train+n_val], t[n_train: n_train+n_val]
    x_test, t_test = x[n_train+n_val:], t[n_train+n_val:]
    
    # Tensorをndarrayに変換
    x_train = x_train.numpy()
    t_train = t_train.numpy()
    x_val = x_val.numpy()
    t_val = t_val.numpy()
    x_test = x_test.numpy()
    t_test = t_test.numpy()
    x_pred = x_pred.numpy()

    # 3次元から2次元に変換
    x_train = x_train.reshape(-1, window*input_dim)
    x_val = x_val.reshape(-1, window*input_dim)
    x_test = x_test.reshape(-1, window*input_dim)
    x_pred = x_pred.reshape(-1, window*input_dim)

    # StandardScaler
    from sklearn.preprocessing import StandardScaler

    # x_train で fit
    x_scaler = StandardScaler()
    x_scaler.fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_val = x_scaler.transform(x_val)
    x_test = x_scaler.transform(x_test)
    x_pred = x_scaler.transform(x_pred)

    # t_train で fit
    t_scaler = StandardScaler()
    t_scaler.fit(t_train)
    t_train = t_scaler.transform(t_train)
    t_val = t_scaler.transform(t_val)
    t_test = t_scaler.transform(t_test)

    # 2次元から3次元に戻す
    x_train = x_train.reshape(-1, window, input_dim)
    x_val = x_val.reshape(-1, window, input_dim)
    x_test = x_test.reshape(-1, window, input_dim)
    x_pred = x_pred.reshape(-1, window, input_dim)

    # CNN用に順番を入替える
    x_train = x_train.transpose(0, 2, 1)
    x_val = x_val.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)
    x_pred = x_pred.transpose(0, 2, 1)

    # ndarray を Tensor に変換
    x_train = torch.tensor(x_train, dtype=torch.float32)
    t_train = torch.tensor(t_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    t_val = torch.tensor(t_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    t_test = torch.tensor(t_test, dtype=torch.float32)
    x_pred = torch.tensor(x_pred, dtype=torch.float32)    
    
    return x_pred, input_dim, t_scaler


# シーケンスデータを作成する関数
def processing_cnn_lstm(df, window):
    
    # df の NaN を削除
    df.dropna(inplace=True)
    
    # df を df_processing に格納
    df_processing = df
    df_x = df_processing.drop(['Adj Close'], axis=1) # 入力データ
    df_t = df_processing[['Adj Close']] # ターゲット
    df_x_columns = df_x.columns # df_x のカラムを取得
    df_x = pd.DataFrame(df_x, columns=df_x_columns) # DataFrame に変換
    df_t = pd.DataFrame(df_t, columns=['Adj Close']) # DataFrame に変換

    # df から後ろのwindowを df_pred に格納
    df_pred = df.tail(window)
    df_pred_x = df_pred.drop(['Adj Close'], axis=1) # 入力データ
    df_pred_x = pd.DataFrame(df_pred_x, columns=df_x_columns) # DataFrame に変換

    # 入力値x, 目標値t
    # ウィンドウサイズ
    window = window
    x, t = [], []

    # 連続するウィンドウを移動させながらループ
    for time in range(len(df_processing) - window):

        # 入力値x：連続する window期間 の 'Adj Close' 以外のデータを取得
        features = df_x.iloc[time:time + window].values
        x.append(features)

        # 目標値t : 'Adj Close' を取得
        target = df_t.iloc[time + window - 1].values
        t.append(target)

    # データをNumPy配列に変換
    x = np.array(x)
    t = np.array(t)

    # df_pred_x を ndarray に変換
    x_pred = np.array(df_pred_x)
    x_pred = x_pred.reshape(-1, window, x_pred.shape[1]) # x_pred を3次元に変換

    # 入力値x の特徴量数
    input_dim = x.shape[2]
    x = x.reshape(-1, window, input_dim) # 入力値xの shape を (batch_size, seq_len) -> (batch_size, seq_len, input_dim) へ変換
    t = t.reshape(-1, 1) # 目標値t shape を (batch_size) -> (batch_size, input_dim) へ変換
    x_pred = x_pred.reshape(-1, window, input_dim) # 予測入力値x_pred の shape を (batch_size, seq_len) -> (batch_size, seq_len, input_dim) へ変換

    # Tensor に変換
    x = torch.tensor(x, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    x_pred = torch.tensor(x_pred, dtype=torch.float32)

    # train, val, test のサンプルサイズを決定
    n_train = int(len(x) * 0.8)
    n_val = int(len(x) * 0.1)
    n_test = len(x) - n_train - n_val

    # train, val, test に分割
    x_train, t_train = x[0: n_train], t[0: n_train]
    x_val, t_val = x[n_train: n_train+n_val], t[n_train: n_train+n_val]
    x_test, t_test = x[n_train+n_val:], t[n_train+n_val:]
    
    # Tensorをndarrayに変換
    x_train = x_train.numpy()
    t_train = t_train.numpy()
    x_val = x_val.numpy()
    t_val = t_val.numpy()
    x_test = x_test.numpy()
    t_test = t_test.numpy()
    x_pred = x_pred.numpy()

    # 3次元から2次元に変換
    x_train = x_train.reshape(-1, window*input_dim)
    x_val = x_val.reshape(-1, window*input_dim)
    x_test = x_test.reshape(-1, window*input_dim)
    x_pred = x_pred.reshape(-1, window*input_dim)

    # MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler

    # x_train で fit
    x_scaler = MinMaxScaler()
    x_scaler.fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_val = x_scaler.transform(x_val)
    x_test = x_scaler.transform(x_test)
    x_pred = x_scaler.transform(x_pred)

    # t_train で fit
    t_scaler = MinMaxScaler()
    t_scaler.fit(t_train)
    t_train = t_scaler.transform(t_train)
    t_val = t_scaler.transform(t_val)
    t_test = t_scaler.transform(t_test)

    # 2次元から3次元に戻す
    x_train = x_train.reshape(-1, window, input_dim)
    x_val = x_val.reshape(-1, window, input_dim)
    x_test = x_test.reshape(-1, window, input_dim)
    x_pred = x_pred.reshape(-1, window, input_dim)

    # CNN用に順番を入替える
    x_train = x_train.transpose(0, 2, 1)
    x_val = x_val.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)
    x_pred = x_pred.transpose(0, 2, 1)

    # ndarray を Tensor に変換
    x_train = torch.tensor(x_train, dtype=torch.float32)
    t_train = torch.tensor(t_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    t_val = torch.tensor(t_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    t_test = torch.tensor(t_test, dtype=torch.float32)
    x_pred = torch.tensor(x_pred, dtype=torch.float32)    
    
    return x_pred, input_dim, t_scaler


# シーケンスデータを作成する関数
def processing_timesnet(df, window):
    
    # df の NaN を削除
    df.dropna(inplace=True)
    
    # df を df_processing に格納
    df_processing = df
    df_x = df_processing.drop(['Adj Close'], axis=1) # 入力データ
    df_t = df_processing[['Adj Close']] # ターゲット
    df_x_columns = df_x.columns # df_x のカラムを取得
    df_x = pd.DataFrame(df_x, columns=df_x_columns) # DataFrame に変換
    df_t = pd.DataFrame(df_t, columns=['Adj Close']) # DataFrame に変換

    # df から後ろのwindowを df_pred に格納
    df_pred = df.tail(window)
    df_pred_x = df_pred.drop(['Adj Close'], axis=1) # 入力データ
    df_pred_x = pd.DataFrame(df_pred_x, columns=df_x_columns) # DataFrame に変換

    # 入力値x, 目標値t
    # ウィンドウサイズ
    window = window
    x, t = [], []

    # 連続するウィンドウを移動させながらループ
    for time in range(len(df_processing) - window):

        # 入力値x：連続する window期間 の 'Adj Close' 以外のデータを取得
        features = df_x.iloc[time:time + window].values
        x.append(features)

        # 目標値t : 'Adj Close' を取得
        target = df_t.iloc[time + window - 1].values
        t.append(target)

    # データをNumPy配列に変換
    x = np.array(x)
    t = np.array(t)

    # df_pred_x を ndarray に変換
    x_pred = np.array(df_pred_x)
    x_pred = x_pred.reshape(-1, window, x_pred.shape[1]) # x_pred を3次元に変換

    # 入力値x の特徴量数
    input_dim = x.shape[2]
    x = x.reshape(-1, window, input_dim) # 入力値xの shape を (batch_size, seq_len) -> (batch_size, seq_len, input_dim) へ変換
    t = t.reshape(-1, 1) # 目標値t shape を (batch_size) -> (batch_size, input_dim) へ変換
    x_pred = x_pred.reshape(-1, window, input_dim) # 予測入力値x_pred の shape を (batch_size, seq_len) -> (batch_size, seq_len, input_dim) へ変換

    # Tensor に変換
    x = torch.tensor(x, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    x_pred = torch.tensor(x_pred, dtype=torch.float32)

    # train, val, test のサンプルサイズを決定
    n_train = int(len(x) * 0.8)
    n_val = int(len(x) * 0.1)
    n_test = len(x) - n_train - n_val

    # train, val, test に分割
    x_train, t_train = x[0: n_train], t[0: n_train]
    x_val, t_val = x[n_train: n_train+n_val], t[n_train: n_train+n_val]
    x_test, t_test = x[n_train+n_val:], t[n_train+n_val:]
    
    # Tensorをndarrayに変換
    x_train = x_train.numpy()
    t_train = t_train.numpy()
    x_val = x_val.numpy()
    t_val = t_val.numpy()
    x_test = x_test.numpy()
    t_test = t_test.numpy()
    x_pred = x_pred.numpy()

    # 3次元から2次元に変換
    x_train = x_train.reshape(-1, window*input_dim)
    x_val = x_val.reshape(-1, window*input_dim)
    x_test = x_test.reshape(-1, window*input_dim)
    x_pred = x_pred.reshape(-1, window*input_dim)

    # StandardScaler
    from sklearn.preprocessing import StandardScaler

    # x_train で fit
    x_scaler = StandardScaler()
    x_scaler.fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_val = x_scaler.transform(x_val)
    x_test = x_scaler.transform(x_test)
    x_pred = x_scaler.transform(x_pred)

    # t_train で fit
    t_scaler = StandardScaler()
    t_scaler.fit(t_train)
    t_train = t_scaler.transform(t_train)
    t_val = t_scaler.transform(t_val)
    t_test = t_scaler.transform(t_test)

    # 2次元から3次元に戻す
    x_train = x_train.reshape(-1, window, input_dim)
    x_val = x_val.reshape(-1, window, input_dim)
    x_test = x_test.reshape(-1, window, input_dim)
    x_pred = x_pred.reshape(-1, window, input_dim)

    # CNN用に順番を入替える
    x_train = x_train.transpose(0, 2, 1)
    x_val = x_val.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)
    x_pred = x_pred.transpose(0, 2, 1)

    # ndarray を Tensor に変換
    x_train = torch.tensor(x_train, dtype=torch.float32)
    t_train = torch.tensor(t_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    t_val = torch.tensor(t_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    t_test = torch.tensor(t_test, dtype=torch.float32)
    x_pred = torch.tensor(x_pred, dtype=torch.float32)    
    
    return x_pred, input_dim, t_scaler


# シーケンスデータを作成する関数
def processing_transformer(df, window):
    
    # df の NaN を削除
    df.dropna(inplace=True)
    
    # df を df_processing に格納
    df_processing = df
    df_x = df_processing.drop(['Adj Close'], axis=1) # 入力データ
    df_t = df_processing[['Adj Close']] # ターゲット
    df_x_columns = df_x.columns # df_x のカラムを取得
    df_x = pd.DataFrame(df_x, columns=df_x_columns) # DataFrame に変換
    df_t = pd.DataFrame(df_t, columns=['Adj Close']) # DataFrame に変換

    # df から後ろのwindowを df_pred に格納
    df_pred = df.tail(window)
    df_pred_x = df_pred.drop(['Adj Close'], axis=1) # 入力データ
    df_pred_x = pd.DataFrame(df_pred_x, columns=df_x_columns) # DataFrame に変換

    # 入力値x, 目標値t
    # ウィンドウサイズ
    window = window
    x, t = [], []

    # 連続するウィンドウを移動させながらループ
    for time in range(len(df_processing) - window):

        # 入力値x：連続する window期間 の 'Adj Close' 以外のデータを取得
        features = df_x.iloc[time:time + window].values
        x.append(features)

        # 目標値t : 'Adj Close' を取得
        target = df_t.iloc[time + window - 1].values
        t.append(target)

    # データをNumPy配列に変換
    x = np.array(x)
    t = np.array(t)

    # df_pred_x を ndarray に変換
    x_pred = np.array(df_pred_x)
    x_pred = x_pred.reshape(-1, window, x_pred.shape[1]) # x_pred を3次元に変換

    # 入力値x の特徴量数
    input_dim = x.shape[2]
    x = x.reshape(-1, window, input_dim) # 入力値xの shape を (batch_size, seq_len) -> (batch_size, seq_len, input_dim) へ変換
    t = t.reshape(-1, 1) # 目標値t shape を (batch_size) -> (batch_size, input_dim) へ変換
    x_pred = x_pred.reshape(-1, window, input_dim) # 予測入力値x_pred の shape を (batch_size, seq_len) -> (batch_size, seq_len, input_dim) へ変換

    # Tensor に変換
    x = torch.tensor(x, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    x_pred = torch.tensor(x_pred, dtype=torch.float32)

    # train, val, test のサンプルサイズを決定
    n_train = int(len(x) * 0.8)
    n_val = int(len(x) * 0.1)
    n_test = len(x) - n_train - n_val

    # train, val, test に分割
    x_train, t_train = x[0: n_train], t[0: n_train]
    x_val, t_val = x[n_train: n_train+n_val], t[n_train: n_train+n_val]
    x_test, t_test = x[n_train+n_val:], t[n_train+n_val:]
    
    # Tensorをndarrayに変換
    x_train = x_train.numpy()
    t_train = t_train.numpy()
    x_val = x_val.numpy()
    t_val = t_val.numpy()
    x_test = x_test.numpy()
    t_test = t_test.numpy()
    x_pred = x_pred.numpy()

    # 3次元から2次元に変換
    x_train = x_train.reshape(-1, window*input_dim)
    x_val = x_val.reshape(-1, window*input_dim)
    x_test = x_test.reshape(-1, window*input_dim)
    x_pred = x_pred.reshape(-1, window*input_dim)

    # StandardScaler
    from sklearn.preprocessing import StandardScaler

    # x_train で fit
    x_scaler = StandardScaler()
    x_scaler.fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_val = x_scaler.transform(x_val)
    x_test = x_scaler.transform(x_test)
    x_pred = x_scaler.transform(x_pred)

    # t_train で fit
    t_scaler = StandardScaler()
    t_scaler.fit(t_train)
    t_train = t_scaler.transform(t_train)
    t_val = t_scaler.transform(t_val)
    t_test = t_scaler.transform(t_test)

    # 2次元から3次元に戻す
    x_train = x_train.reshape(-1, window, input_dim)
    x_val = x_val.reshape(-1, window, input_dim)
    x_test = x_test.reshape(-1, window, input_dim)
    x_pred = x_pred.reshape(-1, window, input_dim)

    # CNN用に順番を入替える
    x_train = x_train.transpose(0, 2, 1)
    x_val = x_val.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)
    x_pred = x_pred.transpose(0, 2, 1)

    # ndarray を Tensor に変換
    x_train = torch.tensor(x_train, dtype=torch.float32)
    t_train = torch.tensor(t_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    t_val = torch.tensor(t_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    t_test = torch.tensor(t_test, dtype=torch.float32)
    x_pred = torch.tensor(x_pred, dtype=torch.float32)    
    
    return x_pred, input_dim, t_scaler