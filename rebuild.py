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


class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=2):
        super().__init__()
        # 初始化嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # 初始化GCN和GAT层
        self.gcn_layers = nn.ModuleList([
            GCNLayer(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        self.gat_layers = nn.ModuleList([
            GATLayer(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])

        # 初始化Actor和Critic网络
        self.actor = Actor(embedding_dim)
        self.critic = Critic(embedding_dim)

        # 渐进式融合的层权重
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, adj_matrix, users, pos_items, neg_items):
        # 初始化嵌入
        num_nodes = adj_matrix.size(0)
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)

        # GCN前向传播
        gcn_embeddings = [all_embeddings]
        for layer in self.gcn_layers:
            h = F.relu(layer(adj_matrix, gcn_embeddings[-1]))
            gcn_embeddings.append(h)

        # GAT前向传播
        gat_embeddings = [all_embeddings]
        for layer in self.gat_layers:
            h = F.relu(layer(adj_matrix, gat_embeddings[-1]))
            gat_embeddings.append(h)

        # 渐进式融合
        layer_weights = F.softmax(self.layer_weights, dim=0)
        fused_emb = 0
        for i in range(1, len(gcn_embeddings)):
            # 拼接GCN和GAT嵌入
            concat_emb = torch.cat([gcn_embeddings[i], gat_embeddings[i]], dim=1)
            # 应用层权重融合
            fused_emb += layer_weights[i - 1] * F.relu(nn.Linear(2 * embedding_dim, embedding_dim)(concat_emb))

        # 分割用户和物品嵌入
        num_users = self.user_embedding.weight.size(0)
        user_emb = fused_emb[:num_users]
        item_emb = fused_emb[num_users:]

        # 获取批次中的嵌入
        u_emb = user_emb[users]
        pos_i_emb = item_emb[pos_items]
        neg_i_emb = item_emb[neg_items]

        # Actor网络计算得分
        pos_score = self.actor(u_emb, pos_i_emb)
        neg_score = self.actor(u_emb, neg_i_emb)

        # Critic网络计算价值
        pos_value = self.critic(u_emb, pos_i_emb)
        neg_value = self.critic(u_emb, neg_i_emb)

        return pos_score, neg_score, pos_value, neg_value