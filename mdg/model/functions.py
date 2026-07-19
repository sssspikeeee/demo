import numpy as np
import torch

import mdg

log = mdg.utils.get_logger()


def batch_graphify(features, lengths, speaker_tensor, wp, wf, edge_type_to_idx, att_model, device):
    node_features, edge_index, edge_norm, edge_type = [], [], [], []
    batch_size = features.size(0)
    length_sum = 0  # 累计节点索引偏移量
    edge_ind = []  # 存储每个样本的边关系
    edge_index_lengths = []  # 存储每个样本的边数量

    # 预计算边关系
    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j].cpu().item(), wp, wf))

    edge_weights = att_model(features, lengths, edge_ind)  # 计算边注意力权重

    for j in range(batch_size):
        cur_len = lengths[j].item()
        # 1. 提取节点特征
        node_features.append(features[j, :cur_len, :])
        # 2. 重新计算边关系
        perms = edge_perms(cur_len, wp, wf)
        # 3. 应用全局索引偏移
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len
        # 4. 记录边数量
        edge_index_lengths.append(len(perms))

        for item, item_rec in zip(perms, perms_rec):
            # 1. 边索引（全局坐标）
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
            # 2. 边权重（注意力分数）
            edge_norm.append(edge_weights[j][item[0], item[1]])
            # edge_norm.append(edge_weights[j, item[0], item[1]])

            # 3. 边类型编码
            speaker_cur = speaker_tensor[j, item[0]].item()
            speaker_nxt = speaker_tensor[j, item[1]].item()
            if item[0] < item[1]:
                c = '0'  # 正向边
            else:
                c = '1'  # 反向边
            edge_type.append(edge_type_to_idx[str(speaker_cur) + str(speaker_nxt) + c])

    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_norm = torch.stack(edge_norm).to(device)  # [E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def edge_perms(length, window_past, window_future):
    """
    Method to construct the edges of a graph (a utterance) considering the past and future window.
    return: list of tuples. tuple -> (vertice(int), neighbor(int))
    """

    all_perms = set()
    array = np.arange(length)
    for j in range(length):   # 遍历每个节点j
        perms = set()  # 当前节点的连接集合

        if window_past == -1 and window_future == -1:
            eff_array = array  # 连接所有节点
        elif window_past == -1:  # use all past context
            eff_array = array[:min(length, j + window_future + 1)]  # 节点连接所有过去的节点和未来窗口内的节点
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past):]  # 节点连接过去窗口内的所有节点和所有未来节点
        else:
            eff_array = array[max(0, j - window_past):min(length, j + window_future + 1)]  # 节点连接过去窗口和未来窗口内的节点

        for item in eff_array:
            perms.add((j, item))  # 添加从 j 到 item 的边   (j, item) 表示从节点j指向节点item的有向边
        all_perms = all_perms.union(perms)  # 将当前节点的边合并到全局集合
    return list(all_perms)
