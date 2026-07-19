import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossModalityAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossModalityAttention, self).__init__()
        self.d_model = d_model

    def forward(self, query, keys, values):
        # query: [batch_size, seq_len_q, d_model]
        # keys: [batch_size, seq_len_k, d_model]
        # values: [batch_size, seq_len_v, d_model]

        scores = torch.matmul(query, keys.transpose(-2, -1)) / (self.d_model ** 0.5)

        weights = nn.functional.softmax(scores, dim=-1)

        output = torch.matmul(weights, values)

        return output


class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.feature_dim = feature_dim

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        Q = self.query(x)  # (batch_size, seq_len, feature_dim)
        K = self.key(x)  # (batch_size, seq_len, feature_dim)
        V = self.value(x)  # (batch_size, seq_len, feature_dim)

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (feature_dim ** 0.5)

        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_out = torch.bmm(attention_weights, V)

        return attention_out


class CrossModalFusion(torch.nn.Module):
    def __init__(self, cross_att_dim, self_att_dim, output_dim):
        super().__init__()
        self.cross_att_dim = cross_att_dim
        self.self_att_dim = self_att_dim
        self.output_dim = output_dim

        self.cross_att = CrossModalityAttention(self.cross_att_dim).to(device)
        self.self_att = SelfAttention(self.self_att_dim).to(device)


    def forward(self, feature1, feature2):
        if len(feature1.shape) == 2:  # (batch_size, feature_dim)
            feature1 = feature1.unsqueeze(1) # (batch_size, 1, feature_dim)
        if len(feature2.shape) == 2:
            feature2 = feature2.unsqueeze(1)

        f1_cro = self.cross_att(feature1, feature2, feature2) # (batch_size, seq_len, feature_dim)
        f2_cro = self.cross_att(feature2, feature1, feature1) # (batch_size, seq_len, feature_dim)

        f1_self = self.self_att(f1_cro.to(device)) # (batch_size, seq_len, feature_dim)
        f2_self = self.self_att(f2_cro.to(device)) # (batch_size, seq_len, feature_dim)

        data_cat = torch.cat([f1_self, f2_self], dim=2) # (batch_size, seq_len, feature_dim*2)
        data_avg = avg_pooling(data_cat.permute(0, 2, 1), self.output_dim).permute(0, 2, 1) # (batch_size, seq_len, output_dim)

        return data_avg

def avg_pooling(features, target_length):
    batch_size, seq_length, feature_dim = features.size()

    pool = nn.AdaptiveAvgPool1d(target_length)

    features = features.transpose(1, 2).reshape(batch_size * feature_dim, seq_length)

    pooled_features = pool(features)

    pooled_features = pooled_features.reshape(batch_size, feature_dim, target_length).transpose(1, 2)

    return pooled_features


if __name__ == "__main__":
    fusion = CrossModalFusion(cross_att_dim=200, self_att_dim=200, output_dim=200)
    total_params = sum(p.numel() for p in fusion.parameters())
    print("total_params:", total_params)

    a_feature = torch.randn(64, 200).to(device)
    v_feature = torch.randn(64, 200).to(device)
    t_feature = torch.randn(64, 200).to(device)
    # a_feature = torch.randn(64, 100, 200).to(device)
    # v_feature = torch.randn(64, 100, 200).to(device)
    # t_feature = torch.randn(64, 100, 200).to(device)
    print('unimodal feature shape:', a_feature.shape)
    print('unimodal feature:', a_feature)

    av_feature = fusion(a_feature, v_feature)
    at_feature = fusion(a_feature, t_feature)
    vt_feature = fusion(v_feature, t_feature)
    print('bimodal feature shape:', av_feature.shape)
    print('bimodal feature:', av_feature)

    avt_feature = fusion(av_feature, t_feature)
    atv_feature = fusion(at_feature, v_feature)
    vta_feature = fusion(vt_feature, a_feature)
    print('trimodal feature shape:', avt_feature.shape)
    print('trimodal feature:', avt_feature)

    fusion_feature = torch.cat([avt_feature, atv_feature, vta_feature], dim=2)  # (B, 1, feature_dim*3)
    print('fusion feature shape:', fusion_feature.shape)
    print('fusion feature:', fusion_feature)