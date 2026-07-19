import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1)  # [T,1]

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1,T,D]

    def forward(self, x):
        """
        x: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:, :T]

class SeqContext(nn.Module):

    def __init__(self, u_dim, g_dim, args):
        super(SeqContext, self).__init__()
        self.input_size = u_dim  # 输入特征维度
        self.hidden_dim = g_dim  # 输出特征维度
        self.use_transformer = False
        if args.rnn == "lstm":
            self.rnn = nn.LSTM(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                               bidirectional=True, num_layers=2, batch_first=True)
        elif args.rnn == "gru":
            self.rnn = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                              bidirectional=True, num_layers=2, batch_first=True)
        elif args.rnn == "transformer":
            self.use_transformer = True
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=self.input_size,
                nhead=4,
                dropout=args.drop_rate,
                batch_first=True,
            )
            self.transformer_encoder = torch.nn.TransformerEncoder(
                encoder_layer, num_layers=2
            )
            self.transformer_out = torch.nn.Linear(
                self.input_size, self.hidden_dim, bias=True
            )

    def forward(self, text_len_tensor, text_tensor):
        if self.use_transformer:
            rnn_out = self.transformer_encoder(text_tensor)
            rnn_out = self.transformer_out(rnn_out)
        else:
            # 打包变长序列
            packed = pack_padded_sequence(
                text_tensor,  # 填充后的输入张量
                text_len_tensor.to('cpu'),  # 实际序列长度（移到CPU）
                batch_first=True,  # 批次维度在前
                enforce_sorted=False  # 不要求按长度排序
            )
            # rnn_out, (_, _) = self.rnn(packed, None)  # BiLSTM
            rnn_out, _ = self.rnn(packed, None)  # BiGRU
            rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)  # 解包序列

        return rnn_out