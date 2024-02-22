import random

import torch
import torch.nn as nn
from torch.distributions import Categorical

from utils import Encoder, self_attention
from collections import OrderedDict
import copy


class RARmodel(nn.Module):
    def __init__(self, batch_size, ques_num, emb_dim, hidden_dim, weigh_dim, target_num,
                 policy_mlp_hidden_dim_list, kt_mlp_hidden_dim_list, use_kt, n_steps, n_head, n_layers, n_ques,
                 device, m=200, rank_num=10):
        super(RARmodel, self).__init__()

        self.batch_size = batch_size
        self.ques_num = ques_num
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.weigh_dim = weigh_dim
        self.use_kt = use_kt
        self.n_steps = n_steps
        self.n_ques = n_ques

        self.ques_embedding = nn.Embedding(ques_num, emb_dim, device=device)
        self.emb_to_hidden = nn.Linear(emb_dim, hidden_dim, device=device)
        self.emb_to_double_hidden = nn.Linear(emb_dim, hidden_dim*2, device=device)

        self.encoder = Encoder(hidden_dim, hidden_dim, n_head=n_head, n_layers=n_layers, drop_prob=0.5).to(device)

        self.kt_mlp = nn.Sequential(
            OrderedDict([
                (
                    'layer{}'.format(i),
                    nn.Linear(kt_mlp_hidden_dim_list[i], kt_mlp_hidden_dim_list[i + 1], device=device)
                 )
                for i in range(len(kt_mlp_hidden_dim_list) - 1)
            ])
        )

        self.ques_correct_to_hidden = nn.Linear(emb_dim+1, hidden_dim).to(device)
        self.init_state_encoder = nn.LSTM(hidden_dim, hidden_dim * 2, batch_first=True).to(device)
        # 推荐过程中跟踪学生状态的LSTM
        self.state_encoder = nn.LSTM(hidden_dim * 2, hidden_dim * 2, batch_first=True).to(device)
        # 序列建模模型
        self.seq_encoder = nn.LSTM(ques_num, hidden_dim).to(device)

        self.W1 = nn.Linear(hidden_dim*2, weigh_dim*2, bias=False).to(device)
        self.W2 = nn.Linear(hidden_dim*2, weigh_dim*2, bias=False).to(device)
        self.W3 = nn.Linear(hidden_dim * 2, weigh_dim * 2, bias=False).to(device)
        self.vt = nn.Linear(weigh_dim*2, 1, bias=False).to(device)
        self.policy_network = nn.Sequential(
            OrderedDict([
                (
                    'layer{}'.format(i),
                    nn.Linear(policy_mlp_hidden_dim_list[i], policy_mlp_hidden_dim_list[i + 1], device=device)
                 )
                for i in range(len(policy_mlp_hidden_dim_list) - 1)
            ])
        )
        self.norm = nn.BatchNorm1d(ques_num).to(device)
        self.sigmoid = nn.Sigmoid()

        self.raw_ques_embedding = None
        self.ques_representation = None
        self.batch_target_emb = None
        self.hc = None
        self.batch_state = None
        self.state_encoder_inputs = None

        # 长为batch_size的向量
        self.last_ques = None
        self.last_n_ques = None

        # 长度为step_num的列表
        self.action_prob_list = []
        self.kt_prob_list = []
        self.action_list = []

        self.device = device

        # 额外补充
        self.target = None
        self.target_num = target_num
        self.batch_target_num = None
        self.batch_T_rep = None
        self.target_table = None
        self.m = m
        self.ranking_loss = 0
        self.rank_num = rank_num

    def forward(self, ques_id, observation):

        action, prob = self.take_action()
        kt_prob = self.step_refresh(ques_id, observation)

        return action, prob, kt_prob

    def initialize(self, exercises_record, targets, batch_size=None):
        targets = copy.deepcopy(targets)

        if not batch_size:
            batch_size = self.batch_size

        self.raw_ques_embedding = None
        self.ques_representation = None
        self.batch_target_emb = None
        self.hc = None
        self.batch_state = None
        self.state_encoder_inputs = None

        self.last_ques = None
        self.last_n_ques = None

        self.kt_prob_list = []
        self.action_prob_list = []
        self.action_list = []
        last_ques_list = []
        last_n_ques_list = []

        self.target_table = torch.zeros(self.batch_size, self.ques_num).to(self.device)
        self.ranking_loss = 0

        # 一、获取ques表征
        self.raw_ques_embedding = self.ques_embedding(torch.arange(self.ques_num).to(self.device))
        ques_embedding = self.emb_to_hidden(self.raw_ques_embedding)

        ques_att_embedding = self.encoder(ques_embedding.unsqueeze(0)).squeeze(0)

        self.ques_representation = torch.cat([ques_att_embedding, ques_embedding], dim=1)

        # 二、获取目标表示
        targets_embedding_list = []
        self.batch_target_num = torch.zeros(self.batch_size).to(self.device)  # batch_size长度向量，记录各学生的学习目标数
        for i in range(len(targets)):  # targets长度不同时，统一成target_num长度，补充元素的值为target_num
            target = targets[i]

            for ques_id in target:
                self.target_table[i, ques_id] = 1

            targets_embedding = self.ques_representation[torch.tensor(list(target)).to(self.device)]
            mean_targets_embedding = torch.mean(targets_embedding, dim=0)
            targets_embedding_list.append(mean_targets_embedding)

            self.batch_target_num[i] = len(targets[i])
            targets[i] = list(targets[i]) + [self.target_num for _ in range(self.target_num - len(targets[i]))]

        self.target = torch.tensor(targets, requires_grad=False).to(self.device)  # batch_size * target_num矩
        self.batch_target_emb = torch.stack(targets_embedding_list).to(self.device)  # batch_size * (hidden_dim*2)

        ques_representation = torch.cat(
            [self.ques_representation, torch.zeros(1, self.hidden_dim * 2).to(self.device)], dim=0)
        self.batch_T_rep = ques_representation[self.target]  # batch_size * target_num * (hidden_dim*2)矩

        # targets_embedding_list = []
        # for target in targets:
        #     targets_embedding = self.ques_representation[torch.tensor(list(target)).to(self.device)]
        #
        #     mean_targets_embedding = torch.mean(targets_embedding, dim=0)
        #     targets_embedding_list.append(mean_targets_embedding)
        #
        # self.batch_target_emb = torch.stack(targets_embedding_list).to(self.device)  # batch_size * (hidden_dim*2)

        batch_init_h = []
        batch_init_c = []

        for exercise_record in exercises_record:
            exercise_record = torch.tensor(exercise_record).to(self.device)
            ques_ids = exercise_record[:, 0]

            last_ques_id = ques_ids[-1]
            try:
                last_n_ques_id = ques_ids[-self.n_ques:]
            except AttributeError:
                last_n_ques_id = ques_ids[-1:]

            last_ques_list.append(last_ques_id.item())
            last_n_ques_list.append(last_n_ques_id)

            raw_ques_embedding = self.raw_ques_embedding[ques_ids]
            corrects = exercise_record[:, 1].view(-1, 1)

            inputs = self.ques_correct_to_hidden(torch.cat([raw_ques_embedding, corrects], dim=1))

            out, (h, c) = self.init_state_encoder(inputs)

            batch_init_h.append(h)
            batch_init_c.append(c)

        self.last_ques = torch.tensor(last_ques_list).to(self.device)

        self.last_n_ques = torch.stack(last_n_ques_list, dim=0)


        batch_init_h = torch.cat(batch_init_h, dim=0).unsqueeze(0)
        batch_init_c = torch.cat(batch_init_c, dim=0).unsqueeze(0)

        self.hc = (batch_init_h, batch_init_c)

        self.state_encoder_inputs = torch.zeros(batch_size, 1, self.hidden_dim*2).to(self.device)

        self.batch_state, self.hc = self.state_encoder(self.state_encoder_inputs, self.hc)

    def take_action(self):

        ques_representation = self.ques_representation + torch.mean(self.ques_representation, dim=0)
        # ques_representation: batch_size * (hidden_dim * 2)
        batch_last_ques_representation = ques_representation[self.last_n_ques].mean(dim=1)
        #
        # batch_target_emb = self.batch_target_emb

        batch_state = self.batch_state.squeeze()

        att_target = self_attention(batch_state.unsqueeze(1), self.batch_T_rep, self.batch_T_rep)

        blend = self.W1(batch_last_ques_representation) + self.W2(att_target.squeeze()) + self.W3(batch_state)

        prob_weigh = self.policy_network(blend)
        prob = prob_weigh.softmax(dim=1)

        sampler = Categorical(prob)
        action = sampler.sample()

        return action, prob

    def step_refresh(self, ques_id, observation):
        ques_id = torch.tensor(ques_id, device=self.device)

        self.last_ques = ques_id
        try:
            if self.last_n_ques.size()[1] < self.n_ques:
                self.last_n_ques = torch.cat([self.last_n_ques, ques_id.unsqueeze(1).to(self.device)], dim=1)
            else:
                self.last_n_ques = torch.cat([self.last_n_ques[:, 1:], ques_id.unsqueeze(1).to(self.device)], dim=1)
        except AttributeError:
            if self.last_n_ques.size()[1] < 1:
                self.last_n_ques = torch.cat([self.last_n_ques, ques_id.unsqueeze(1).to(self.device)], dim=1)
            else:
                self.last_n_ques = torch.cat([self.last_n_ques[:, 1:], ques_id.unsqueeze(1).to(self.device)], dim=1)

        raw_ques_embedding = self.raw_ques_embedding[ques_id]

        corrects = torch.tensor(observation, device=self.device).view(-1, 1)

        inputs = self.ques_correct_to_hidden(torch.cat([raw_ques_embedding, corrects], dim=1))
        out, self.hc = self.init_state_encoder(inputs.unsqueeze(1), self.hc)

    def get_kt_prob(self, ques_id):
        ques_id = torch.tensor(ques_id, device=self.device)
        self.last_ques = ques_id
        prob = None

        self.state_encoder_inputs = self.ques_representation[ques_id].unsqueeze(1)

        self.batch_state, hc = self.state_encoder(self.state_encoder_inputs, self.hc)

        if self.use_kt:
            prob = self.sigmoid(self.kt_mlp(self.batch_state).squeeze())

        return prob

    def get_ranking_loss(self, seq):  # seq: step_num * batch_size * ques_dim
        seq_rep, _ = self.seq_encoder(seq)  # step_num * batch_size * hidden_dim
        seq_rep = seq_rep[-1]  # 只取最后一步的最终结果  batch_size * hidden_dim

        for i in range(self.batch_size):
            index = torch.randint(0, self.batch_size, (self.rank_num,))

            # 学习目标间的差距
            target_dist = (self.target_table[i] - self.target_table[index]).abs().sum(1)  # rank_num长度向量

            # 路径表示上的差距
            pdist = nn.PairwiseDistance(p=2)
            seq_dist = pdist(seq_rep[i], seq_rep[index])  # rank_num长度向量

            # 计算loss
            m_tensor = torch.full((self.rank_num,), self.m).to(self.device)
            ranking_loss = torch.clamp(m_tensor * target_dist - seq_dist, min=0, max=100000).mean()

            self.ranking_loss += ranking_loss

        return self.ranking_loss / self.batch_size
