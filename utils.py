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

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()

        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        score = F.softmax(score, dim=3)

        v = score @ v

        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, n_head, dropout_rate):
        super(MultiHeadAttention, self).__init__()

        self.model_dim = model_dim
        self.n_head = n_head
        self.head_dim = self.model_dim

        self.linear_k = nn.Linear(self.model_dim, self.head_dim * self.n_head)
        self.linear_v = nn.Linear(self.model_dim, self.head_dim * self.n_head)
        self.linear_q = nn.Linear(self.model_dim, self.head_dim * self.n_head)

        self.linear_final = nn.Linear(self.head_dim * self.n_head, self.model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.scaled_dot_product_attention = ScaleDotProductAttention()

    def forward(self, inputs, mask=None):
        q = self.linear_q(inputs)
        k = self.linear_k(inputs)
        v = self.linear_v(inputs)
        batch_size = k.size()[0]

        q_ = q.view(batch_size, self.n_head, -1, self.head_dim)
        k_ = k.view(batch_size, self.n_head, -1, self.head_dim)
        v_ = v.view(batch_size, self.n_head, -1, self.head_dim)

        context, _ = self.scaled_dot_product_attention(q_, k_, v_, mask)
        output = context.transpose(1, 2) .contiguous().view(batch_size, -1, self.n_head * self.head_dim)
        output = self.linear_final(output)
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
        _x = x
        x = self.attention(x, mask=s_mask)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(hidden_dim=input_dim,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask=None):

        for layer in self.layers:
            x = layer(x, s_mask)

        return x

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

    score = torch.matmul(W1(q), W2(k).transpose(1, 2)) / math.sqrt(hidden_dim)
    score = F.softmax(score, dim=2)
    v = torch.matmul(score, W3(v))
    return v
