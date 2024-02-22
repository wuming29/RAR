import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import math
import modules
from torch.autograd import Variable
import pickle
import random



class dkt(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.node_dim = args.dim
        self.concept_num = args.concept_num
        self.max_concept = args.max_concepts
        self.device = args.device
        self.predictor = modules.funcs(args.n_layer, args.dim, args.problem_number, args.dropout).to(args.device) #args.dim +
        self.gru_h = modules.mygru(0, args.dim * 1, args.dim).to(args.device)
        self.seq_length = args.seq_len
        self.prob_emb = nn.Parameter(torch.randn(args.problem_number * 2, args.dim).to(args.device), requires_grad=True)

        self.h = None

        showi0 = []
        for i in range(0, 20000):
            showi0.append(i)
        self.show_index = torch.tensor(showi0).to(args.device)
        self.sigmoid = torch.nn.Sigmoid()
        self.ones = torch.tensor(1).to(args.device)
        self.zeros = torch.tensor(0).to(args.device)

    def cell(self, h, this_input):  # this_input是第i个序列节点的200位学生的信息
        prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, operate = this_input
        filter0 = torch.where(related_concept_index == 0, torch.tensor(0.0).to(self.device), torch.tensor(1.0).to(self.device))
        # data_len = prob_ids.size()[0]
        data_len = prob_ids.size()[0]
        first_idnex = self.show_index[0: data_len]
        total_pre = self.predictor(h)
        prob = self.predictor(h)[first_idnex, prob_ids].unsqueeze(-1)  # 每个学生在当前序列位置处回答当前序列位置处的问题时的回答正确预测值 (200*1:200个学生，当前序列位置)

        # response_emb = self.prob_emb[prob_ids * 2 + operate.long().squeeze(1)]
        response_emb = self.prob_emb[prob_ids * 2 + (self.sigmoid(prob)>0.5).int().squeeze(1)]

        next_p_state = self.gru_h(response_emb, h)
        return next_p_state, prob, total_pre

    def forward(self, inputs):

        probs = []
        total_pre = []
        data_len = len(inputs[0])  # 该batch学生数
        seq_len = len(inputs[1])  # 序列长度
        h = torch.zeros(data_len, self.node_dim).to(self.device)
        # for i in range(0, self.seq_length):
        for i in range(0, seq_len):
            h, prob, total_pre = self.cell(h, inputs[1][i])  # inputs[1][i]是第i个序列节点的200位学生的信息，prob：batch_size*1矩
            probs.append(prob)  # probs：长度为seq_len列表，每个元素记录了该batch所有学生回答当前序列位置题目的答对概率

        prob_tensor = torch.cat(probs, dim=1)  # batch_size*seq_len矩，每行是一个确定学生的回答各问题的答对概率

        predict, pre_hiddenstates = [], []
        seq_num = inputs[0]

        # constrain_pdkt = 0
        for i in range(0, data_len):
            this_prob = prob_tensor[i][0: seq_num[i]]
            predict.append(this_prob)

        return torch.cat(predict, dim=0), total_pre

    def reset(self, batch_size=1):
        self.h = torch.zeros(batch_size, 64).to(self.device)
        with torch.no_grad():
            total_pre = self.predictor(self.h)  # 对所有问题，未激活的预测
            total_probability = self.sigmoid(total_pre)  # 所有问题激活后的预测概率值

        return self.h, total_probability

    def step(self, h, batch_problem_id):  # batch_problem_id: batch_size长度向量
        step_sigmoid = torch.nn.Sigmoid().to(self.device)
        batch_idx = torch.tensor([i for i in range(len(batch_problem_id))]).to(self.device)
        self.h = h if h is not None else self.h

        with torch.no_grad():
            total_pre = self.predictor(self.h)  # batch_size * ques_num矩：对所有问题，未激活的预测
            total_probability = step_sigmoid(total_pre)  # batch_size * ques_num矩：所有问题激活后的预测概率值
            this_probability = total_probability[batch_idx, batch_problem_id]  # batch_size向量：当前输入问题的预测概率值
            total_observation = (total_probability > 0.5).int()  # batch_size * ques_num矩：阈值0.5，获得所有题目答对的0\1预测值
            batch_observation = total_observation[batch_idx, batch_problem_id]  # batch_size向量：阈值0.5，获得当前题目答对的0\1预测值
            batch_score_pre = torch.sum(total_probability, dim=1)  # batch_size长的向量：回答问题前的总得分
            # self.prob_emb是(args.problem_number * 2) * args.dim的矩阵
            response_emb = self.prob_emb[batch_problem_id * 2 + batch_observation]  # batch_size * arg.dim(64)矩
            # response_emb = self.prob_emb[problem_id * 2 + y]
            self.h = self.gru_h(response_emb, self.h)

            total_probability = step_sigmoid(self.predictor(self.h))
            batch_score_aft = torch.sum(total_probability, dim=1)  # batch_size长的向量：回答问题后的总得分
            batch_reward = batch_score_aft - batch_score_pre  # batch_size长的向量：回答问题后的进步分

        return self.h, batch_observation, batch_reward, batch_score_aft

    @property
    def total_probability(self):
        step_sigmoid = torch.nn.Sigmoid()

        with torch.no_grad():
            total_pre = self.predictor(self.h)  # 对所有问题，未激活的预测
            total_probability = step_sigmoid(total_pre)  # 所有问题激活后的预测概率值

        return total_probability

    @property
    def score(self):
        step_sigmoid = torch.nn.Sigmoid()

        with torch.no_grad():
            total_pre = self.predictor(self.h)  # 对所有问题，未激活的预测
            total_probability = step_sigmoid(total_pre)  # 所有问题激活后的预测概率值
            score = total_probability.sum().item()  # 回答问题前的总得分

        return score

    # target: batch_size * self.target_num矩
    def test_target_score(self, target):
        batch_size = target.size()[0]
        target_num = target.size()[1]
        index = torch.arange(0, batch_size).unsqueeze(dim=1).repeat(1, target_num).to(self.device)
        # batch_size * self.target_num矩, 各元素是该学生在该target下的得分
        target_score = self.total_probability[index, target]
        # batch长的向量，表示各学生学习目标总得分
        target_score = target_score.sum(dim=1)

        return target_score

    # def step_valid(self, h, problem_id, y):
    #     step_sigmoid = torch.nn.Sigmoid()
    #     reward = 0
    #
    #     with torch.no_grad():
    #         total_pre = self.predictor(h)  # 对所有问题，未激活的预测
    #         total_probability = step_sigmoid(total_pre)  # 所有问题激活后的预测概率值
    #         this_probability = total_probability[0][problem_id]  # 当前输入问题的预测概率值
    #         total_observation = (total_probability > 0.5).int()  #  阈值0.5，获得所有题目答对的0\1预测值
    #         observation = total_observation[0][problem_id]  #  阈值0.5，获得当前题目答对的0\1预测值
    #         for i in range(len(total_probability[0])):
    #             reward += float(total_probability[0][i])
    #         # response_emb = self.prob_emb[problem_id * 2 + observation.long()]
    #         response_emb = self.prob_emb[problem_id * 2 + y]
    #         h = self.gru_h(response_emb, h)
    #
    #
    #
    #     return h, observation, reward, total_probability