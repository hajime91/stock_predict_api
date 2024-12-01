from data.df_stock_data import df_stock_data_n225, df_stock_data_nikkei_cme, df_stock_data_dow, df_stock_data_usd_jpy, df_stock_data_gold, df_stock_data_wti, df_stock_data_us_10yb, df_stock_data_sp500, df_stock_data_nasdaq
#from data.defs.def_processing_lstm import processing_lstm
#from data.defs.def_processing_cnn import processing_cnn
#from data.defs.def_processing_cnn_lstm import processing_cnn_lstm
#from data.defs.def_processing_timesnet import processing_timesnet
#from data.defs.def_processing_transformer import processing_transformer
from data.defs.def_processing_model import processing_lstm, processing_cnn, processing_cnn_lstm, processing_timesnet, processing_transformer

#from df_stock_data import df_stock_data_n225, df_stock_data_nikkei_cme, df_stock_data_dow, df_stock_data_usd_jpy, df_stock_data_gold, df_stock_data_wti, df_stock_data_us_10yb
#from defs.def_processing_lstm import processing_lstm
#from defs.def_processing_cnn import processing_cnn
#from defs.def_processing_cnn_lstm import processing_cnn_lstm
#from defs.def_processing_timesnet import processing_timesnet
#from defs.def_processing_transformer import processing_transformer

# ウィンドウサイズ
window = 30

# nikkei
nikkei_x_pred_lstm, nikkei_input_dim_lstm, nikkei_t_scaler_lstm = processing_lstm(df_stock_data_n225, window)
nikkei_x_pred_cnn, nikkei_input_dim_cnn, nikkei_t_scaler_cnn = processing_cnn(df_stock_data_n225, window)
nikkei_x_pred_cnn_lstm, nikkei_input_dim_cnn_lstm, nikkei_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_n225, window)
nikkei_x_pred_timesnet, nikkei_input_dim_timesnet, nikkei_t_scaler_timesnet = processing_timesnet(df_stock_data_n225, window)
nikkei_x_pred_transformer, nikkei_input_dim_transformer, nikkei_t_scaler_transformer = processing_transformer(df_stock_data_n225, window)

# nikkei_cme
nikkei_cme_x_pred_lstm, nikkei_cme_input_dim_lstm, nikkei_cme_t_scaler_lstm = processing_lstm(df_stock_data_nikkei_cme, window)
nikkei_cme_x_pred_cnn, nikkei_cme_input_dim_cnn, nikkei_cme_t_scaler_cnn = processing_cnn(df_stock_data_nikkei_cme, window)
nikkei_cme_x_pred_cnn_lstm, nikkei_cme_input_dim_cnn_lstm, nikkei_cme_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_nikkei_cme, window)
nikkei_cme_x_pred_timesnet, nikkei_cme_input_dim_timesnet, nikkei_cme_t_scaler_timesnet = processing_timesnet(df_stock_data_nikkei_cme, window)
nikkei_cme_x_pred_transformer, nikkei_cme_input_dim_transformer, nikkei_cme_t_scaler_transformer = processing_transformer(df_stock_data_nikkei_cme, window)

# dow
dow_x_pred_lstm, dow_input_dim_lstm, dow_t_scaler_lstm = processing_lstm(df_stock_data_dow, window)
dow_x_pred_cnn, dow_input_dim_cnn, dow_t_scaler_cnn = processing_cnn(df_stock_data_dow, window)
dow_x_pred_cnn_lstm, dow_input_dim_cnn_lstm, dow_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_dow, window)
dow_x_pred_timesnet, dow_input_dim_timesnet, dow_t_scaler_timesnet = processing_timesnet(df_stock_data_dow, window)
dow_x_pred_transformer, dow_input_dim_transformer, dow_t_scaler_transformer = processing_transformer(df_stock_data_dow, window)

# usd_jpy
usd_jpy_x_pred_lstm, usd_jpy_input_dim_lstm, usd_jpy_t_scaler_lstm = processing_lstm(df_stock_data_usd_jpy, window)
usd_jpy_x_pred_cnn, usd_jpy_input_dim_cnn, usd_jpy_t_scaler_cnn = processing_cnn(df_stock_data_usd_jpy, window)
usd_jpy_x_pred_cnn_lstm, usd_jpy_input_dim_cnn_lstm, usd_jpy_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_usd_jpy, window)
usd_jpy_x_pred_timesnet, usd_jpy_input_dim_timesnet, usd_jpy_t_scaler_timesnet = processing_timesnet(df_stock_data_usd_jpy, window)
usd_jpy_x_pred_transformer, usd_jpy_input_dim_transformer, usd_jpy_t_scaler_transformer = processing_transformer(df_stock_data_usd_jpy, window)

# gold
gold_x_pred_lstm, gold_input_dim_lstm, gold_t_scaler_lstm = processing_lstm(df_stock_data_gold, window)
gold_x_pred_cnn, gold_input_dim_cnn, gold_t_scaler_cnn = processing_cnn(df_stock_data_gold, window)
gold_x_pred_cnn_lstm, gold_input_dim_cnn_lstm, gold_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_gold, window)
gold_x_pred_timesnet, gold_input_dim_timesnet, gold_t_scaler_timesnet = processing_timesnet(df_stock_data_gold, window)
gold_x_pred_transformer, gold_input_dim_transformer, gold_t_scaler_transformer = processing_transformer(df_stock_data_gold, window)

# wti
wti_x_pred_lstm, wti_input_dim_lstm, wti_t_scaler_lstm = processing_lstm(df_stock_data_wti, window)
wti_x_pred_cnn, wti_input_dim_cnn, wti_t_scaler_cnn = processing_cnn(df_stock_data_wti, window)
wti_x_pred_cnn_lstm, wti_input_dim_cnn_lstm, wti_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_wti, window)
wti_x_pred_timesnet, wti_input_dim_timesnet, wti_t_scaler_timesnet = processing_timesnet(df_stock_data_wti, window)
wti_x_pred_transformer, wti_input_dim_transformer, wti_t_scaler_transformer = processing_transformer(df_stock_data_wti, window)

# us_10yb
us_10yb_x_pred_lstm, us_10yb_input_dim_lstm, us_10yb_t_scaler_lstm = processing_lstm(df_stock_data_us_10yb, window)
us_10yb_x_pred_cnn, us_10yb_input_dim_cnn, us_10yb_t_scaler_cnn = processing_cnn(df_stock_data_us_10yb, window)
us_10yb_x_pred_cnn_lstm, us_10yb_input_dim_cnn_lstm, us_10yb_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_us_10yb, window)
us_10yb_x_pred_timesnet, us_10yb_input_dim_timesnet, us_10yb_t_scaler_timesnet = processing_timesnet(df_stock_data_us_10yb, window)
us_10yb_x_pred_transformer, us_10yb_input_dim_transformer, us_10yb_t_scaler_transformer = processing_transformer(df_stock_data_us_10yb, window)

# sp500
sp500_x_pred_lstm, sp500_input_dim_lstm, sp500_t_scaler_lstm = processing_lstm(df_stock_data_sp500, window)
sp500_x_pred_cnn, sp500_input_dim_cnn, sp500_t_scaler_cnn = processing_cnn(df_stock_data_sp500, window)
sp500_x_pred_cnn_lstm, sp500_input_dim_cnn_lstm, sp500_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_sp500, window)
sp500_x_pred_timesnet, sp500_input_dim_timesnet, sp500_t_scaler_timesnet = processing_timesnet(df_stock_data_sp500, window)
sp500_x_pred_transformer, sp500_input_dim_transformer, sp500_t_scaler_transformer = processing_transformer(df_stock_data_sp500, window)

# nasdaq
nasdaq_x_pred_lstm, nasdaq_input_dim_lstm, nasdaq_t_scaler_lstm = processing_lstm(df_stock_data_nasdaq, window)
nasdaq_x_pred_cnn, nasdaq_input_dim_cnn, nasdaq_t_scaler_cnn = processing_cnn(df_stock_data_nasdaq, window)
nasdaq_x_pred_cnn_lstm, nasdaq_input_dim_cnn_lstm, nasdaq_t_scaler_cnn_lstm = processing_cnn_lstm(df_stock_data_nasdaq, window)
nasdaq_x_pred_timesnet, nasdaq_input_dim_timesnet, nasdaq_t_scaler_timesnet = processing_timesnet(df_stock_data_nasdaq, window)
nasdaq_x_pred_transformer, nasdaq_input_dim_transformer, nasdaq_t_scaler_transformer = processing_transformer(df_stock_data_nasdaq, window)

# 各データセットとモデルごとにx_predをまとめる関数
def get_x_preds():
    return {
        "nikkei": {
            "LSTM": nikkei_x_pred_lstm,
            "CNN": nikkei_x_pred_cnn,
            "CNN-LSTM": nikkei_x_pred_cnn_lstm,
            "TimesNet": nikkei_x_pred_timesnet,
            "Transformer": nikkei_x_pred_transformer
        },
        "nikkei_cme": {
            "LSTM": nikkei_cme_x_pred_lstm,
            "CNN": nikkei_cme_x_pred_cnn,
            "CNN-LSTM": nikkei_cme_x_pred_cnn_lstm,
            "TimesNet": nikkei_cme_x_pred_timesnet,
            "Transformer": nikkei_cme_x_pred_transformer
        },
        "dow": {
            "LSTM": dow_x_pred_lstm,
            "CNN": dow_x_pred_cnn,
            "CNN-LSTM": dow_x_pred_cnn_lstm,
            "TimesNet": dow_x_pred_timesnet,
            "Transformer": dow_x_pred_transformer
        },
        "usdjpy": {
            "LSTM": usd_jpy_x_pred_lstm,
            "CNN": usd_jpy_x_pred_cnn,
            "CNN-LSTM": usd_jpy_x_pred_cnn_lstm,
            "TimesNet": usd_jpy_x_pred_timesnet,
            "Transformer": usd_jpy_x_pred_transformer
        },
        "gold": {
            "LSTM": gold_x_pred_lstm,
            "CNN": gold_x_pred_cnn,
            "CNN-LSTM": gold_x_pred_cnn_lstm,
            "TimesNet": gold_x_pred_timesnet,
            "Transformer": gold_x_pred_transformer
        },
        "wti": {
            "LSTM": wti_x_pred_lstm,
            "CNN": wti_x_pred_cnn,
            "CNN-LSTM": wti_x_pred_cnn_lstm,
            "TimesNet": wti_x_pred_timesnet,
            "Transformer": wti_x_pred_transformer
        },
        "us_10yb": {
            "LSTM": us_10yb_x_pred_lstm,
            "CNN": us_10yb_x_pred_cnn,
            "CNN-LSTM": us_10yb_x_pred_cnn_lstm,
            "TimesNet": us_10yb_x_pred_timesnet,
            "Transformer": us_10yb_x_pred_transformer
        },
        "sp500": {
            "LSTM": sp500_x_pred_lstm,
            "CNN": sp500_x_pred_cnn,
            "CNN-LSTM": sp500_x_pred_cnn_lstm,
            "TimesNet": sp500_x_pred_timesnet,
            "Transformer": sp500_x_pred_transformer
        },
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
    return {
        "nikkei": {
            "LSTM": nikkei_input_dim_lstm,
            "CNN": nikkei_input_dim_cnn,
            "CNN-LSTM": nikkei_input_dim_cnn_lstm,
            "TimesNet": nikkei_input_dim_timesnet,
            "Transformer": nikkei_input_dim_transformer
        },
        "nikkei_cme": {
            "LSTM": nikkei_cme_input_dim_lstm,
            "CNN": nikkei_cme_input_dim_cnn,
            "CNN-LSTM": nikkei_cme_input_dim_cnn_lstm,
            "TimesNet": nikkei_cme_input_dim_timesnet,
            "Transformer": nikkei_cme_input_dim_transformer
        },
        "dow": {
            "LSTM": dow_input_dim_lstm,
            "CNN": dow_input_dim_cnn,
            "CNN-LSTM": dow_input_dim_cnn_lstm,
            "TimesNet": dow_input_dim_timesnet,
            "Transformer": dow_input_dim_transformer
        },
        "usdjpy": {
            "LSTM": usd_jpy_input_dim_lstm,
            "CNN": usd_jpy_input_dim_cnn,
            "CNN-LSTM": usd_jpy_input_dim_cnn_lstm,
            "TimesNet": usd_jpy_input_dim_timesnet,
            "Transformer": usd_jpy_input_dim_transformer
        },
        "gold": {
            "LSTM": gold_input_dim_lstm,
            "CNN": gold_input_dim_cnn,
            "CNN-LSTM": gold_input_dim_cnn_lstm,
            "TimesNet": gold_input_dim_timesnet,
            "Transformer": gold_input_dim_transformer
        },
        "wti": {
            "LSTM": wti_input_dim_lstm,
            "CNN": wti_input_dim_cnn,
            "CNN-LSTM": wti_input_dim_cnn_lstm,
            "TimesNet": wti_input_dim_timesnet,
            "Transformer": wti_input_dim_transformer
        },
        "us_10yb": {
            "LSTM": us_10yb_input_dim_lstm,
            "CNN": us_10yb_input_dim_cnn,
            "CNN-LSTM": us_10yb_input_dim_cnn_lstm,
            "TimesNet": us_10yb_input_dim_timesnet,
            "Transformer": us_10yb_input_dim_transformer
        },
        "sp500": {
            "LSTM": sp500_input_dim_lstm,
            "CNN": sp500_input_dim_cnn,
            "CNN-LSTM": sp500_input_dim_cnn_lstm,
            "TimesNet": sp500_input_dim_timesnet,
            "Transformer": sp500_input_dim_transformer
        },
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
    return {
        "nikkei": {
            "LSTM": nikkei_t_scaler_lstm,
            "CNN": nikkei_t_scaler_cnn,
            "CNN-LSTM": nikkei_t_scaler_cnn_lstm,
            "TimesNet": nikkei_t_scaler_timesnet,
            "Transformer": nikkei_t_scaler_transformer
        },
        "nikkei_cme": {
            "LSTM": nikkei_cme_t_scaler_lstm,
            "CNN": nikkei_cme_t_scaler_cnn,
            "CNN-LSTM": nikkei_cme_t_scaler_cnn_lstm,
            "TimesNet": nikkei_cme_t_scaler_timesnet,
            "Transformer": nikkei_cme_t_scaler_transformer
        },
        "dow": {
            "LSTM": dow_t_scaler_lstm,
            "CNN": dow_t_scaler_cnn,
            "CNN-LSTM": dow_t_scaler_cnn_lstm,
            "TimesNet": dow_t_scaler_timesnet,
            "Transformer": dow_t_scaler_transformer
        },
        "usdjpy": {
            "LSTM": usd_jpy_t_scaler_lstm,
            "CNN": usd_jpy_t_scaler_cnn,
            "CNN-LSTM": usd_jpy_t_scaler_cnn_lstm,
            "TimesNet": usd_jpy_t_scaler_timesnet,
            "Transformer": usd_jpy_t_scaler_transformer
        },
        "gold": {
            "LSTM": gold_t_scaler_lstm,
            "CNN": gold_t_scaler_cnn,
            "CNN-LSTM": gold_t_scaler_cnn_lstm,
            "TimesNet": gold_t_scaler_timesnet,
            "Transformer": gold_t_scaler_transformer
        },
        "wti": {
            "LSTM": wti_t_scaler_lstm,
            "CNN": wti_t_scaler_cnn,
            "CNN-LSTM": wti_t_scaler_cnn_lstm,
            "TimesNet": wti_t_scaler_timesnet,
            "Transformer": wti_t_scaler_transformer
        },
        "us_10yb": {
            "LSTM": us_10yb_t_scaler_lstm,
            "CNN": us_10yb_t_scaler_cnn,
            "CNN-LSTM": us_10yb_t_scaler_cnn_lstm,
            "TimesNet": us_10yb_t_scaler_timesnet,
            "Transformer": us_10yb_t_scaler_transformer
        },
        "sp500": {
            "LSTM": sp500_t_scaler_lstm,
            "CNN": sp500_t_scaler_cnn,
            "CNN-LSTM": sp500_t_scaler_cnn_lstm,
            "TimesNet": sp500_t_scaler_timesnet,
            "Transformer": sp500_t_scaler_transformer
        },
        "nasdaq": {
            "LSTM": nasdaq_t_scaler_lstm,
            "CNN": nasdaq_t_scaler_cnn,
            "CNN-LSTM": nasdaq_t_scaler_cnn_lstm,
            "TimesNet": nasdaq_t_scaler_timesnet,
            "Transformer": nasdaq_t_scaler_transformer
        }
    }
