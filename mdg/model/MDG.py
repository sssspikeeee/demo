import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .EdgeAtt import EdgeAtt
from .GNN import GNN
from .Predictor import Predictor
from .functions import batch_graphify
from .CrossModalFusion import CrossModalFusion
from .GCA  import GCA
from .STMA import SymmetricTriModalAttention
import mdg

log = mdg.utils.get_logger()

import math


class MulMoAttn(nn.Module):
    def __init__(self, in_channels, args):
        super(MulMoAttn, self).__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** args.scale_para
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


class MDG(nn.Module):

    def __init__(self, args):
        super(MDG, self).__init__()
        self.args = args
        u_dim_a, u_dim_v, u_dim_t = 100, 2048, 768
        g_dim = 200
        # g_dim = 210
        h1_dim = 100
        h2_dim = 100
        hc_dim = 100
        tag_size = 1

        # n_head = 1
        # n_head = 2
        # n_head = 3
        n_head = 4
        # n_head = 5
        # n_head = 6

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.rnn_a = SeqContext(u_dim_a, g_dim, args)
        self.rnn_v = SeqContext(u_dim_v, g_dim, args)
        self.rnn_t = SeqContext(u_dim_t, g_dim, args)
        self.CMFM = CrossModalFusion(g_dim, g_dim, g_dim)
        self.mmattn = MulMoAttn(g_dim, args)
        self.GCA = GCA(g_dim, n_head)
        self.STMA = SymmetricTriModalAttention(g_dim, n_head)
        self.iAFF = SeqIAFF(dim=g_dim*3, r=4, dropout=args.drop_rate, use_layernorm=True)
        self.HSTMA = HierarchicalSymmetricTriModalAttention(g_dim, n_head)
        
        self.edge_att = EdgeAtt(g_dim*3, args)
        self.gnn = GNN(g_dim*3, h1_dim, h2_dim, n_head, args)

        self.pred = Predictor(g_dim*3 + h2_dim*n_head, hc_dim, tag_size, args) # for RGCN+GT+GSAGE and GCN+GT+GSAGE and RGCN+GT

        # self.pred = Predictor(g_dim*3 + h2_dim, hc_dim, tag_size, args) # for RGCN+GSAGE
        # self.pred = Predictor(g_dim*3 + h1_dim, hc_dim, tag_size, args) # for only RGCN or GSAGE or FC
        # self.pred = Predictor(g_dim*3 + h1_dim*n_head, hc_dim, tag_size, args) # for only GT and GT+GSAGE

        # self.pred = Predictor(g_dim*3 + h2_dim, hc_dim, tag_size, args)
        # self.pred = Predictor(g_dim*3, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)


    def get_rep(self, data):
        node_features_T = self.rnn_t(data["text_len_tensor"], data["text_tensor"]) # [batch_size, mx_len, D_g]
        node_features_A = self.rnn_a(data["text_len_tensor"], data["audio_tensor"]) # [batch_size, mx_len, D_g]
        node_features_V = self.rnn_v(data["text_len_tensor"], data["visual_tensor"]) # [batch_size, mx_len, D_g]

        # 拼接
        # node_features = torch.cat((node_features_T, node_features_A, node_features_V), 2)

        '''
        # CA
        node_features_A_att = self.mmattn(node_features_V, node_features_A)
        node_features_V_att = self.mmattn(node_features_A, node_features_V)
        node_features_T_att = self.mmattn(node_features_T, node_features_T)
        node_features = torch.cat((node_features_T_att, node_features_A_att, node_features_V_att), 2)

        # 多层CA
        node_features_AV = self.CMFM(node_features_A, node_features_V)
        node_features_VT = self.CMFM(node_features_V, node_features_T)
        node_features_AT = self.CMFM(node_features_A, node_features_T)
        node_features = torch.cat((node_features_AV, node_features_VT, node_features_AT), 2)
        node_features_AVT = self.CMFM(node_features_AV, node_features_T)
        node_features_VTA = self.CMFM(node_features_VT, node_features_A)
        node_features_ATV = self.CMFM(node_features_AT, node_features_V)
        node_features = torch.cat((node_features_AVT, node_features_VTA, node_features_ATV), 2)

        # GCA
        node_features = self.GCA(node_features_A, node_features_V, node_features_T)
        '''
    
        # STMA
        node_features_newT, node_features_newA, node_features_newV = self.STMA(node_features_T, node_features_A, node_features_V)
        # node_features_newT, node_features_newA, node_features_newV = self.HSTMA(node_features_T, node_features_A, node_features_V)
        node_features = torch.cat((node_features_newT, node_features_newA, node_features_newV), 2)




        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)

        # print('features shape:', features.shape)
        graph_out = self.gnn(features, edge_index, edge_norm, edge_type)
        # print('graph_out shape:', graph_out.shape)

        return graph_out, features

    def forward(self, data):
        graph_out, features = self.get_rep(data)
        reprentation = torch.cat([features, graph_out], dim=-1)
        # print('reprentation shape:', reprentation.shape)
        out = self.pred(reprentation, data["text_len_tensor"])

        # out = self.pred(torch.cat([features], dim=-1), data["text_len_tensor"])
        
        return out

    def get_loss(self, data):
        graph_out, features = self.get_rep(data)
        loss = self.pred.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["text_len_tensor"])

        # loss = self.pred.get_loss(torch.cat([features], dim=-1),
        #                          data["label_tensor"], data["text_len_tensor"])
        

        return loss
