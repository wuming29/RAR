import torch
import torch.nn as nn
import modules



class dkt(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.node_dim = args.dim
        self.concept_num = args.concept_num
        self.max_concept = args.max_concepts
        self.device = args.device
        self.predictor = modules.funcs(args.n_layer, args.dim, args.problem_number, args.dropout).to(args.device)
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

    def cell(self, h, this_input):
        prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, operate = this_input
        data_len = prob_ids.size()[0]
        first_idnex = self.show_index[0: data_len]
        total_pre = self.predictor(h)
        prob = self.predictor(h)[first_idnex, prob_ids].unsqueeze(-1)

        response_emb = self.prob_emb[prob_ids * 2 + (self.sigmoid(prob)>0.5).int().squeeze(1)]

        next_p_state = self.gru_h(response_emb, h)
        return next_p_state, prob, total_pre

    def forward(self, inputs):

        probs = []
        total_pre = []
        data_len = len(inputs[0])
        seq_len = len(inputs[1])
        h = torch.zeros(data_len, self.node_dim).to(self.device)
        for i in range(0, seq_len):
            h, prob, total_pre = self.cell(h, inputs[1][i]) 
            probs.append(prob) 

        prob_tensor = torch.cat(probs, dim=1)

        predict, pre_hiddenstates = [], []
        seq_num = inputs[0]

        for i in range(0, data_len):
            this_prob = prob_tensor[i][0: seq_num[i]]
            predict.append(this_prob)

        return torch.cat(predict, dim=0), total_pre

    def reset(self, batch_size=1):
        self.h = torch.zeros(batch_size, 64).to(self.device)
        with torch.no_grad():
            total_pre = self.predictor(self.h)
            total_probability = self.sigmoid(total_pre) 
        return self.h, total_probability

    def step(self, h, batch_problem_id):
        step_sigmoid = torch.nn.Sigmoid().to(self.device)
        batch_idx = torch.tensor([i for i in range(len(batch_problem_id))]).to(self.device)
        self.h = h if h is not None else self.h

        with torch.no_grad():
            total_pre = self.predictor(self.h)
            total_probability = step_sigmoid(total_pre) 
            total_observation = (total_probability > 0.5).int() 
            batch_observation = total_observation[batch_idx, batch_problem_id] 
            batch_score_pre = torch.sum(total_probability, dim=1)
            response_emb = self.prob_emb[batch_problem_id * 2 + batch_observation]
            self.h = self.gru_h(response_emb, self.h)

            total_probability = step_sigmoid(self.predictor(self.h))
            batch_score_aft = torch.sum(total_probability, dim=1) 
            batch_reward = batch_score_aft - batch_score_pre

        return self.h, batch_observation, batch_reward, batch_score_aft

    @property
    def total_probability(self):
        step_sigmoid = torch.nn.Sigmoid()

        with torch.no_grad():
            total_pre = self.predictor(self.h)
            total_probability = step_sigmoid(total_pre) 

        return total_probability

    @property
    def score(self):
        step_sigmoid = torch.nn.Sigmoid()

        with torch.no_grad():
            total_pre = self.predictor(self.h) 
            total_probability = step_sigmoid(total_pre)  
            score = total_probability.sum().item() 

        return score

    def test_target_score(self, target):
        batch_size = target.size()[0]
        target_num = target.size()[1]
        index = torch.arange(0, batch_size).unsqueeze(dim=1).repeat(1, target_num).to(self.device)
        target_score = self.total_probability[index, target]
        target_score = target_score.sum(dim=1)

        return target_score
