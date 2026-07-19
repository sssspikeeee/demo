import torch.nn as nn
from torch_geometric.nn import RGCNConv, GraphConv, GATConv, TransformerConv, SAGEConv, GCNConv
import argparse

'''
class GNN(nn.Module):

    def __init__(self, g_dim, h1_dim, h2_dim, n_head, args):
        super(GNN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        # self.conv2 = GraphConv(h1_dim, h2_dim)
        # self.conv2 = GATConv(h1_dim, h2_dim, n_head, 0.6)
        self.conv2 = TransformerConv(h1_dim, h2_dim, n_head, 0.6)



    def forward(self, node_features, edge_index, edge_norm, edge_type):
        # x = self.conv1(node_features, edge_index, edge_type, edge_norm=edge_norm)
        x = self.conv1(node_features, edge_index, edge_type)
        x = self.conv2(x, edge_index)

        return x
'''


class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, n_head, args):
        super(GNN, self).__init__()
        self.args = args
        self.num_relations = 2 * args.n_speakers ** 2

        
        # ablation
        '''
        self.rgcn = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        # self.bn1 = nn.BatchNorm1d(h1_dim)
        self.a1_gtf = TransformerConv(g_dim, h1_dim, heads=n_head, concat=True)
        self.a1_gsage = SAGEConv(g_dim, h1_dim)

        self.gcn = GCNConv(g_dim, h1_dim)
        self.wognn = nn.Linear(g_dim, h1_dim)

        # self.conv2 = GATConv(h1_dim, h2_dim, n_head, 0.6)
        # self.conv2 = TransformerConv(h1_dim, h2_dim, n_head, 0.6)
        self.gtf = TransformerConv(h1_dim, h2_dim, heads=n_head, concat=True)
        # self.bn2 = nn.BatchNorm1d(h2_dim * n_head)
        self.h2_dim_transformer = h2_dim * n_head  # 记录 transformer 的输出维度

        input_dim = self.h2_dim_transformer
        # self.conv3 = SAGEConv(input_dim, h2_dim)
        self.gsage = SAGEConv(input_dim, input_dim)
        self.a2_gsage = SAGEConv(h1_dim, h2_dim)
        # self.bn3 = nn.BatchNorm1d(h2_dim)
        '''


        # Full GNN RGCN+GT+GSAGE
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        self.bn1 = nn.BatchNorm1d(h1_dim)
        self.conv2 = TransformerConv(h1_dim, h2_dim, heads=n_head, concat=True)
        self.bn2 = nn.BatchNorm1d(h2_dim * n_head)
        self.h2_dim_transformer = h2_dim * n_head
        input_dim = self.h2_dim_transformer
        self.conv3 = SAGEConv(input_dim, input_dim)
        self.bn3 = nn.BatchNorm1d(h2_dim)



    def forward(self, node_features, edge_index, edge_norm, edge_type):
        # residual = node_features  # Initial residual for the first layer

        '''
        # w/GNN RGCN+GT+GSAGE
        x = self.rgcn(node_features, edge_index, edge_type)
        # x = nn.functional.leaky_relu(self.bn1(x))
        # if x.size(1) != residual.size(1):
        #     residual = nn.functional.pad(residual, (0, x.size(1) - residual.size(1)))
        # x = x + residual  # Residual connection
        # residual = x  # Update residual for the second layer

        x = self.gtf(x, edge_index)
        # x = nn.functional.leaky_relu(self.bn2(x))
        # if x.size(1) != residual.size(1):
        #     residual = nn.functional.pad(residual, (0, x.size(1) - residual.size(1)))
        # x = x + residual  # Residual connection
        # residual = x  # Update residual

        x = self.gsage(x, edge_index)
        # x = nn.functional.leaky_relu(self.bn3(x))
        # if x.size(1) != residual.size(1):
        #     residual = nn.functional.pad(residual, (0, x.size(1) - residual.size(1)))
        # x = x + residual  # Residual connection
        '''

        # Full GNN RGCN+GT+GSAGE
        x = self.conv1(node_features, edge_index, edge_type)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)



        '''
        # w/GNN GCN+GT+GSAGE
        x = self.gcn(node_features, edge_index)
        x = self.gtf(x, edge_index)
        x = self.gsage(x, edge_index)
        '''

        '''
        # w/GNN RGCN+GT+_
        x = self.rgcn(node_features, edge_index, edge_type)
        x = self.gtf(x, edge_index)
        '''

        '''
        # w/GNN RGCN+_+GSAGE
        x = self.rgcn(node_features, edge_index, edge_type)
        x = self.a2_gsage(x, edge_index)
        '''

        '''
        # w/GNN _+GT+GSAGE
        x = self.a1_gtf(node_features, edge_index)
        x = self.gsage(x, edge_index)
        '''

        '''
        # w/GNN only RGCN or GT or GSAGE
        # x = self.rgcn(node_features, edge_index, edge_type)
        # x = self.a1_gsage(node_features, edge_index)
        x = self.a1_gtf(node_features, edge_index)
        '''

        '''
        # w/o GNN
        x = self.wognn(node_features)
        '''

        return x


if __name__ == "__main__":
    args = argparse.Namespace()
    args.n_speakers = 2
    model = GNN(g_dim=600, h1_dim=100, h2_dim=100, n_head=4, args=args)
    print(model)
    params = sum(p.numel() for p in model.parameters())
    print(params)