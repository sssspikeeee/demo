import torch
from torch import nn

class MulMoAttn(nn.Module):
    def __init__(self, in_channels):
        super(MulMoAttn, self).__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** (-0.5)
        self.attend = nn.Softmax(dim=-1)

        self.linear_k.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_q.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2. / in_channels))

    def forward(self, y, x):
        query = self.linear_q(y)
        key = self.linear_k(x)
        value = self.linear_v(x)
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, value)
        return out

class GatedCrossModalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        # head-wise gate
        self.gate = nn.Linear(d_model, n_heads)
        self.n_heads = n_heads

    def forward(self, query, key, value):
        """
        query: [B, Nq, D]   (主模态)
        key/value: [B, Nk, D] (辅模态)
        """

        attn_out, _ = self.attn(query, key, value)
        # attn_out: [B, Nq, D]

        # compute gate (query-dependent)
        # gate = torch.sigmoid(self.gate(query))  # [B, Nq, H]

        # # expand gate to head dimensions
        # gate = gate.repeat_interleave(
        #     attn_out.size(-1) // self.n_heads,
        #     dim=-1
        # )  # [B, Nq, D]

        # return attn_out * gate
        return attn_out

class TriModalGatedFusion(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.GCA = GatedCrossModalAttention(d_model, n_heads)

    def forward(self, x_m1, x_m2, x_m3):
        """
        x_m1:   [B, N1, D]
        x_m2:  [B, N2, D]
        x_m3: [B, N3, D]
        """

        m1_m2 = self.GCA(
            query=x_m1,
            key=x_m2,
            value=x_m2
        )

        m1_m3 = self.GCA(
            query=x_m1,
            key=x_m3,
            value=x_m3
        )

        z = x_m1 + m1_m2 + m1_m3
        # print("Gated fusion output shape:", z.shape)
        return z

class GCA(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.tri_modal_fusion = TriModalGatedFusion(d_model, n_heads)
    
    def forward(self, x_a, x_v, x_t):
        """
        x_a: audio features [B, N, D]
        x_v: visual features [B, N, D]
        x_t: text features [B, N, D]
        """
        if len(x_a.shape) == 2:  # (batch_size, feature_dim)
            x_a = x_a.unsqueeze(1) # (batch_size, 1, feature_dim)
        if len(x_v.shape) == 2:
            x_v = x_v.unsqueeze(1)
        if len(x_t.shape) == 2:
            x_t = x_t.unsqueeze(1)

        fused_a = self.tri_modal_fusion(x_a, x_v, x_t)
        fused_v = self.tri_modal_fusion(x_v, x_a, x_t)
        fused_t = self.tri_modal_fusion(x_t, x_a, x_v)
        fused_features = torch.cat([fused_a, fused_v, fused_t], dim=2)  # (B, N, feature_dim*3)
        if len(x_a.shape) == 2 and len(x_v.shape) == 2 and len(x_t.shape) == 2:
            fused_features = fused_features.squeeze(1)  # (B, feature_dim*3)
        return fused_features

if __name__ == "__main__":
    # 测试GCA模块
    batch_size = 64
    seq_len = 1
    feature_dim = 200
    n_heads = 4

    x_a = torch.randn(batch_size, feature_dim)  # audio features
    x_v = torch.randn(batch_size, feature_dim)  # visual features
    x_t = torch.randn(batch_size, feature_dim)  # text features
    print("Input feature shapes:", x_a.shape, x_v.shape, x_t.shape)

    gca = GCA(d_model=feature_dim, n_heads=n_heads)
    fused_features = gca(x_a, x_v, x_t)
    print("Fused features shape:", fused_features.shape)  # 应该是 (batch_size, seq_len, feature_dim*3)