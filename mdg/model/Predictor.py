import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mdg

log = mdg.utils.get_logger()


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_size, tag_size, args):
        super(Predictor, self).__init__()
        self.emotion_att = MaskedSegAtt(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.drop = nn.Dropout(args.drop_rate)
        self.lin2 = nn.Linear(hidden_size, tag_size)
        self.relu = nn.ReLU()
        # self.mse_loss = nn.MSELoss()
        self.utt_len = np.load('./res/utt_len.npy')
        self.device = args.device

    def get_score(self, h, text_len_tensor):
        h_hat = self.emotion_att(h, text_len_tensor)
        hidden = self.drop(F.relu(self.lin1(h_hat)))
        # hidden = self.drop(F.relu(self.lin1(h)))
        scores = self.relu(self.lin2(hidden))
        
        return scores

    def forward(self, h, text_len_tensor):
        log_prob = self.get_score(h, text_len_tensor)
        # y_hat = torch.argmax(log_prob, dim=-1)

        return log_prob

    '''
    原始CCC计算方法
    def CCCLoss(self, x, y):
        # y = torch.concat(y, 0).to(self.args.device)
        # x = torch.concat(x, 0).to(self.args.device)
        x_utt = []
        y_utt = []

        y = torch.reshape(y, (-1,))
        x = torch.reshape(x, (-1,))

        for i in range(len(self.utt_len)-1):
            y_utt.append(y[self.utt_len[i]+1])
            st, et = self.utt_len[i]+1, self.utt_len[i]+1+self.utt_len[i+1]
            x_utt.append(torch.mean(x[st:et]))

        x_utt = torch.tensor(x_utt).to(self.device)
        y_utt = torch.tensor(y_utt).to(self.device)

        # x_utt = torch.stack(x_utt).to(self.device)
        # y_utt = torch.stack(y_utt).to(self.device)

        ccc = 2*torch.cov(torch.concat([x_utt, y_utt], 0)) / (x_utt.var() + y_utt.var() + (x_utt.mean() - y_utt.mean())**2)

        return 1-ccc
    '''

    # L1: MSE Loss
    def loss_mse(self, y_pred, y_true):
        return nn.MSELoss()(y_pred, y_true)
    
    # L2: CCC Loss
    def loss_ccc(self, y_pred, y_true, eps=1e-8):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        mean_pred = torch.mean(y_pred)
        mean_true = torch.mean(y_true)

        var_pred = torch.var(y_pred, unbiased=False)
        var_true = torch.var(y_true, unbiased=False)

        cov = torch.mean((y_pred - mean_pred) * (y_true - mean_true))

        ccc = (2 * cov) / (var_pred + var_true +
                        (mean_pred - mean_true) ** 2 + eps)
        return 1.0 - ccc

    # Additional Variance Loss
    def variance_loss(self, y_pred, y_true):
        """
        Variance consistency loss
        """
        std_pred = torch.std(y_pred, unbiased=False)
        std_true = torch.std(y_true, unbiased=False)

        return torch.abs(std_pred - std_true)

    # L3: CCC + MSE
    def loss_ccc_mse(self, y_pred, y_true): 
        return self.loss_mse(y_pred, y_true) + self.loss_ccc(y_pred, y_true) 
    # L4: CCC + MSE + Variance
    def loss_ccc_mse_var(self, y_pred, y_true): 
        return self.loss_mse(y_pred, y_true) + self.loss_ccc(y_pred, y_true) + self.variance_loss(y_pred, y_true)
    
    # L5: Weighted CCC + MSE
    def loss_weighted_ccc_mse(self, y_pred, y_true, alpha=0.5): 
        return alpha * self.loss_mse(y_pred, y_true) + (1 - alpha) * self.loss_ccc(y_pred, y_true)
    # L6: Weighted CCC + MSE + Variance
    def loss_weighted_ccc_mse_var(self, y_pred, y_true, lambda_mse=1.0, lambda_ccc=1.0, lambda_var=1.0): 
        return lambda_mse * self.loss_mse(y_pred, y_true) + lambda_ccc * self.loss_ccc(y_pred, y_true) + lambda_var * self.variance_loss(y_pred, y_true)

    
    def get_loss(self, h, label_tensor, text_len_tensor):
        label_tensor = torch.reshape(label_tensor, (-1, 1))
        score_pred = self.get_score(h, text_len_tensor)

        '''
        loss1 = self.mse_loss(score_pred, label_tensor)
        # print(score_pred.shape, label_tensor.shape)
        # loss2 = self.CCCLoss(torch.mean(score_pred, 1), torch.mean(label_tensor, 1))
        loss2 = self.CCCLoss(score_pred, label_tensor)
        loss = loss1 + loss2
        '''
        # loss = self.loss_ccc(score_pred, label_tensor)
        # loss = self.loss_mse(score_pred, label_tensor)
        

        # loss = self.loss_ccc_mse(score_pred, label_tensor)
        loss = self.loss_ccc_mse_var(score_pred, label_tensor)

        # loss = self.loss_weighted_ccc_mse(score_pred, label_tensor, alpha=0.6)
        # loss = self.loss_weighted_ccc_mse_var(score_pred, label_tensor, lambda_mse=1.0, lambda_ccc=1.0, lambda_var=1.0)

        # Ablation Study
        # loss = self.loss_weighted_ccc_mse_var(score_pred, label_tensor, lambda_mse=1.0, lambda_ccc=1.0, lambda_var=0.0) # CCC + MSE
        # loss = self.loss_weighted_ccc_mse_var(score_pred, label_tensor, lambda_mse=0.0, lambda_ccc=1.0, lambda_var=1.0) # CCC + Variance
        # loss = self.loss_weighted_ccc_mse_var(score_pred, label_tensor, lambda_mse=0.0, lambda_ccc=1.0, lambda_var=0.0) # CCC
        # loss = self.loss_weighted_ccc_mse_var(score_pred, label_tensor, lambda_mse=1.0, lambda_ccc=0.0, lambda_var=1.0) # MSE + Variance
        # loss = self.loss_weighted_ccc_mse_var(score_pred, label_tensor, lambda_mse=1.0, lambda_ccc=0.0, lambda_var=0.0) # MSE
        # loss = self.loss_weighted_ccc_mse_var(score_pred, label_tensor, lambda_mse=0.0, lambda_ccc=0.0, lambda_var=1.0) # Variance

        return loss



class MaskedSegAtt(nn.Module):
    """
    掩码分段注意力机制
    功能：对批次中的每个样本单独计算自注意力，只考虑有效序列部分
    特点：避免填充部分干扰，为每个序列学习内部依赖关系
    """
    def __init__(self, input_dim):
        super(MaskedSegAtt, self).__init__()
        self.lin = nn.Linear(input_dim, input_dim)

    def forward(self, h, text_len_tensor):
        batch_size = text_len_tensor.size(0)
        x = self.lin(h)  # [node_num, H] - 线性变换后的特征
        ret = torch.zeros_like(h)  # 初始化输出张量
        s = 0  # 节点指针，用于跟踪批次中的位置
        for bi in range(batch_size):
            cur_len = text_len_tensor[bi].item()  # 当前样本的实际长度
            y = x[s: s + cur_len]  # 当前样本的变换后特征 [L, H]
            z = h[s: s + cur_len]  # 当前样本的原始特征 [L, H]
            scores = torch.mm(z, y.t())  # [L, L] - 计算自注意力分数
            probs = F.softmax(scores, dim=1)
            out = z.unsqueeze(0) * probs.unsqueeze(-1)  # [1, L, H] x [L, L, 1] --> [L, L, H]
            out = torch.sum(out, dim=1)  # [L, H] - 对第二个维度(L)求和
            ret[s: s + cur_len, :] = out  # 将当前样本结果填充到输出
            s += cur_len

        return ret


