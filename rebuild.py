import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim)

    def forward(self, adj, x):
        # 图卷积操作: (D^-1/2 A D^-1/2) X W
        x = torch.spmm(adj, x)  # 稀疏矩阵乘法
        return self.weight(x)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(2 * out_dim, 1)

    def forward(self, adj, x):
        # 计算注意力系数
        h = self.fc(x)
        N = h.size(0)

        # 构建所有节点对的特征
        h_repeat = h.repeat_interleave(N, dim=0)
        h_tile = h.repeat(N, 1)
        a_input = torch.cat([h_repeat, h_tile], dim=1)

        # 计算注意力得分
        e = F.leaky_relu(self.attn(a_input).squeeze())
        attn = e.view(N, N) * adj.to_dense()  # 只保留邻接节点的注意力
        attn = F.softmax(attn, dim=1)

        # 应用注意力更新节点特征
        return torch.matmul(attn, h)


class Actor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, user_emb, item_emb):
        # 计算用户-物品分数
        user_emb = self.fc(user_emb)
        return torch.sum(user_emb * item_emb, dim=1)  # 内积计算得分


class Critic(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, user_emb, item_emb):
        # 评估当前用户-物品对的价值
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.mlp(x)