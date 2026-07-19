import torch
import torch.nn as nn
import math

class MulMoAttn(nn.Module):
    def __init__(self, in_channels):
        super(MulMoAttn, self).__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels)
        self.linear_k = nn.Linear(in_channels, in_channels)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** (-0.5)
        self.attend = nn.Softmax(dim=-1)

        self.linear_k.weight.data.normal_(0, math.sqrt(2. / in_channels))
        self.linear_q.weight.data.normal_(0, math.sqrt(2. / in_channels))
        self.linear_v.weight.data.normal_(0, math.sqrt(2. / in_channels))

    def forward(self, y, x):
        query = self.linear_q(y)
        key = self.linear_k(x)
        value = self.linear_v(x)
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, value)
        return out

class SymmetricTriModalAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mmattn = MulMoAttn(dim)

        self.gate = nn.ModuleDict({
            "T_A": nn.Linear(dim, dim),
            "T_V": nn.Linear(dim, dim),
            "A_T": nn.Linear(dim, dim),
            "A_V": nn.Linear(dim, dim),
            "V_T": nn.Linear(dim, dim),
            "V_A": nn.Linear(dim, dim),
        })

        self.norm = nn.LayerNorm(dim)

    def gated_attn(self, Q, K, V, gate_layer):
        # C, _ = self.attn(Q, K, V)
        C = self.mmattn(Q, K)
        # gate = torch.sigmoid(gate_layer(Q))
        # return C * gate
        return C

    def forward(self, H_t, H_a, H_v):
        # Text updates
        T_A = self.gated_attn(H_t, H_a, H_a, self.gate["T_A"])
        T_V = self.gated_attn(H_t, H_v, H_v, self.gate["T_V"])
        H_t_new = H_t + T_A + T_V
        # H_t_new = self.norm(H_t_new)

        # Audio updates
        A_T = self.gated_attn(H_a, H_t, H_t, self.gate["A_T"])
        A_V = self.gated_attn(H_a, H_v, H_v, self.gate["A_V"])
        H_a_new = H_a + A_T + A_V
        # H_a_new = self.norm(H_a_new)

        # Video updates
        V_T = self.gated_attn(H_v, H_t, H_t, self.gate["V_T"])
        V_A = self.gated_attn(H_v, H_a, H_a, self.gate["V_A"])
        H_v_new = H_v + V_T + V_A
        # H_v_new = self.norm(H_v_new)

        return H_t_new, H_a_new, H_v_new

if __name__ == "__main__":
    model = SymmetricTriModalAttention(dim=100)
    print(model)
    params = sum(p.numel() for p in model.parameters())
    print(params)
