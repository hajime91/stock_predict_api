import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# LSTM
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size=50, num_layers=2, dropout_prob=0.2, lr=0.001, weight_decay=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)  # 出力層の調整
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # (batch_size, hidden_size) -> (batch_size, 1)
        return out


# CNN
class CNNRegressor(pl.LightningModule):
    def __init__(self, input_dim, num_filters=64, kernel_size=3, dropout_prob=0.2, lr=0.001, weight_decay=0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # グローバルプーリング層を追加
        self.fc = nn.Linear(num_filters, 1)  # 出力は1つのタイムステップ
        self.dropout = nn.Dropout(dropout_prob)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.global_pool(x)  # グローバルプーリングを適用
        x = x.view(x.size(0), -1)  # (バッチサイズ, フィルタ数)
        x = self.fc(x)
        return x


# CNN-LSTM
class CNN_LSTM_Regressor(pl.LightningModule):
    def __init__(self, input_dim, num_filters=64, kernel_size=3, hidden_size=50, num_layers=2, dropout_prob=0.2, lr=0.001, weight_decay=0):
        super().__init__()

        # CNN部分
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # LSTM部分
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout_prob)

        # 出力層
        self.fc = nn.Linear(hidden_size, 1)  # 4つのタイムステップの予測
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        # CNNの計算
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)

        # LSTMの計算
        x = x.permute(0, 2, 1)  # LSTMの入力に合わせてshape変換 (バッチサイズ, シーケンス長, 特徴量数)
        out, (h, c) = self.lstm(x)

        # 出力層
        out = self.fc(out[:, -1, :])  # LSTMの最後の出力を使う
        return out


# TimesNet
class TimesNetRegressor(pl.LightningModule):

    def __init__(self, input_dim, hidden_size=50, num_layers=2, dropout_prob=0.2, lr=0.001, weight_decay=0):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.weight_decay = weight_decay

        # 入力データの正規化層
        self.batch_norm = nn.BatchNorm1d(input_dim)

        # TimesNet層の構築
        layers = []
        in_channels = input_dim
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_channels = hidden_size

        self.temporal_blocks = nn.Sequential(*layers)

        # 出力層
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 入力データの正規化
        x = self.batch_norm(x)

        # TimesNetで特徴量抽出
        out = self.temporal_blocks(x)

        # 平均プーリング
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)

        # 出力層
        out = self.fc(out)
        return out


# Transformer
class TransformerRegressor(pl.LightningModule):

    def __init__(self, input_dim, hidden_size=50, num_layers=2, num_heads=4, dropout_prob=0.2, lr=0.001, weight_decay=0):
        super().__init__()

        # Transformer層の構築
        self.embedding = nn.Linear(input_dim, hidden_size)  # 入力の埋め込み層（線形変換）

        # Transformerのエンコーダ層
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # self.transformer = nn.Transformer(d_model=hidden_size,
        #                                   nhead=num_heads,
        #                                   num_encoder_layers=num_layers,
        #                                   num_decoder_layers=num_layers,
        #                                   dropout=dropout_prob)

        self.fc = nn.Linear(hidden_size, 1)  # 出力層

        # 学習率
        self.lr = lr

        # L2正則化
        self.weight_decay = weight_decay

    def forward(self, x):
        # 入力の埋め込み（線形変換）
        x = x.permute(2, 0, 1)
        x = self.embedding(x)

        # Transformerを通す
        out = self.transformer_encoder(x)  # 入力とターゲットが同じなので、xを2回渡す

        # 最後の時刻の出力を全結合層に入力
        out = self.fc(out[-1, :, :])  # 最後の時刻の出力を取り出し、(batch_size, hidden_size) -> (batch_size, 1)
        return out