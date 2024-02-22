import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, hidden_list):
        super(MLP, self).__init__()

        self.model = nn.Sequential(OrderedDict([('layer{}'.format(i), nn.Linear(hidden_list[i], hidden_list[i+1]))
                                                for i in range(len(hidden_list) - 1)]))


# https://zhuanlan.zhihu.com/p/127030939
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    # 输入qkv矩阵，输出注意力权值及经过注意力权值加权后的v。
    # qkv均为batch_size * n_head（多头数） * n（项目数）* head_dim（qkv每个向量的长度，该值qkv相同）维矩阵
    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()  # length: 项目数

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)  # batch_size * n_head * n * n

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        # n*n矩阵中第i行第j列代表第i个项目的q查询第j个项目的k值。这里对第i个项目查询各个k后的，关于第k项的权值做softmax，即每行的和为1.
        score = F.softmax(score, dim=3)

        # 4. multiply with Value
        # 第i行是第i个项目查询过所有项目后得到的输出向量
        v = score @ v  # batch_size * n_head * n * head_dim

        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, n_head, dropout_rate):  # 每个项目的表示向量长度为model_dim
        super(MultiHeadAttention, self).__init__()

        self.model_dim = model_dim
        self.n_head = n_head
        self.head_dim = self.model_dim  # 每个头的qkv向量长度也为项目的表示长度

        self.linear_k = nn.Linear(self.model_dim, self.head_dim * self.n_head)
        self.linear_v = nn.Linear(self.model_dim, self.head_dim * self.n_head)
        self.linear_q = nn.Linear(self.model_dim, self.head_dim * self.n_head)

        self.linear_final = nn.Linear(self.head_dim * self.n_head, self.model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.scaled_dot_product_attention = ScaleDotProductAttention()

    def forward(self, inputs, mask=None):  # inputs: batch_size * n(n个项目) * model_dim(每个项目的表示向量长度)
        q = self.linear_q(inputs)
        k = self.linear_k(inputs)
        v = self.linear_v(inputs)
        batch_size = k.size()[0]

        q_ = q.view(batch_size, self.n_head, -1, self.head_dim)  # -1一项的值为n(n个项目)
        k_ = k.view(batch_size, self.n_head, -1, self.head_dim)
        v_ = v.view(batch_size, self.n_head, -1, self.head_dim)

        context, _ = self.scaled_dot_product_attention(q_, k_, v_, mask)  # context: batch_size * n_head * n * head_dim
        # output: batch_size * n_head * n * head_dim
        #      => batch_size * n * n_head * head_dim
        #      => batch_size * n * (n_head * head_dim)
        output = context.transpose(1, 2) .contiguous().view(batch_size, -1, self.n_head * self.head_dim)
        output = self.linear_final(output)  # => batch_size * n * model_dim
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):

    def __init__(self, model_dim, hidden, drop_prob=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(model_dim, hidden)
        self.linear2 = nn.Linear(hidden, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, hidden_dim, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim=hidden_dim, n_head=n_head, dropout_rate=drop_prob)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = FeedForward(model_dim=hidden_dim, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, s_mask):
        _x = x  # _x: batch_size * n(n个项目) * model_dim(每个项目的表示向量长度)
        x = self.attention(x, mask=s_mask)  # x: batch_size * n * model_dim
        x = self.norm1(x + _x)  # 残差网络并层归一化（输出的每个向量内部各元素做归一化）
        x = self.dropout1(x)

        # 残差加归一化后的输出再过全连接+残差
        _x = x  # 原始输出
        x = self.ffn(x)  # 输出过全连接
        x = self.norm2(x + _x)  # 加残差并归一
        x = self.dropout2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(hidden_dim=input_dim,
                                                  ffn_hidden=ffn_hidden,  # 自注意力后的全连接是双层的，中间隐层的维度
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask=None):

        for layer in self.layers:
            x = layer(x, s_mask)

        return x

# q: batch_size*n*q_input_dim. k: batch_size*item_num*k_input_dim. v:batch_size*item_num*v_input_dim
def self_attention(q, k, v, hidden_dim=None, value_dim=None):
    q_input_dim = q.shape[-1]
    k_input_dim = k.shape[-1]
    v_input_dim = v.shape[-1]
    if not hidden_dim:
        hidden_dim = q_input_dim
    if not value_dim:
        value_dim = v_input_dim

    device = q.device
    W1 = nn.Linear(q_input_dim, hidden_dim, device=device)
    W2 = nn.Linear(k_input_dim, hidden_dim, device=device)
    W3 = nn.Linear(v_input_dim, value_dim, device=device)

    score = torch.matmul(W1(q), W2(k).transpose(1, 2)) / math.sqrt(hidden_dim)  # batch_size*n*item_num
    score = F.softmax(score, dim=2)  # batch_size*n*item_num
    v = torch.matmul(score, W3(v))  # batch_size*n*value_dim
    return v
