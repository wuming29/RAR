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
from torch.distributions import Categorical
from tqdm import tqdm


class dkt(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.node_dim = args.dim
        self.concept_num = args.concept_num
        self.max_concept = args.max_concepts
        self.device = args.device
        self.predictor = modules.funcs(args.n_layer, args.dim, args.problem_number, args.dropout)
        self.gru_h = modules.mygru(0, args.dim * 1, args.dim)

        self.seq_length = args.seq_len
        
        self.prob_emb = nn.Parameter(torch.randn(args.problem_number * 2, args.dim).to(args.device), requires_grad=True)

        showi0 = []
        for i in range(0, 20000):
            showi0.append(i)
        self.show_index = torch.tensor(showi0).to(args.device)
        self.sigmoid = torch.nn.Sigmoid()
        self.ones = torch.tensor(1).to(args.device)
        self.zeros = torch.tensor(0).to(args.device)
     
    def cell(self, h, this_input):
        prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, operate = this_input
        filter0 = torch.where(related_concept_index == 0, torch.tensor(0.0).to(self.device), torch.tensor(1.0).to(self.device))
        data_len = prob_ids.size()[0]
        first_idnex = self.show_index[0: data_len]
        prob = self.predictor(h)[first_idnex, prob_ids].unsqueeze(-1)

        response_emb = self.prob_emb[prob_ids * 2 + operate.long().squeeze(1)]

        next_p_state = self.gru_h(response_emb, h)
        return next_p_state, prob

    def forward(self, inputs):

        probs = []
        data_len = len(inputs[0])
        h = torch.zeros(data_len, self.node_dim).to(self.device)
        for i in range(0, self.seq_length):
            h, prob = self.cell(h, inputs[1][i])
            probs.append(prob) 

        prob_tensor = torch.cat(probs, dim = 1)

        predict, pre_hiddenstates = [], []
        seq_num = inputs[0]

        # constrain_pdkt = 0
        for i in range(0, data_len):
            this_prob = prob_tensor[i][0 : seq_num[i]]
            predict.append(this_prob)

        return torch.cat(predict, dim = 0), 0

class iekt_c(nn.Module): 

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gamma = args.gamma
        self.lamb = args.lamb
        self.node_dim = args.dim
        self.concept_num = args.concept_num
        self.max_concept = args.max_concepts
        self.device = args.device
        self.seq_length = args.seq_len
        self.predictor = modules.funcs(args.n_layer, args.dim * 5, 1, args.dropout)
        self.cog_matrix = nn.Parameter(torch.randn(args.cog_levels, args.dim * 2).to(args.device), requires_grad=True) 
        self.acq_matrix = nn.Parameter(torch.randn(args.acq_levels, args.dim * 2).to(args.device), requires_grad=True)
        self.select_preemb = modules.funcs(args.n_layer, args.dim * 3, args.cog_levels, args.dropout) 
        self.checker_emb = modules.funcs(args.n_layer, args.dim * 12, args.acq_levels, args.dropout) 
        self.prob_emb = nn.Parameter(torch.randn(args.problem_number - 1, args.dim).to(args.device), requires_grad=True)
        self.gru_h = modules.mygru(0, args.dim * 4, args.dim)
        showi0 = []
        for i in range(0, 10000):
            showi0.append(i)
        self.show_index = torch.tensor(showi0).to(args.device)
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num - 1, args.dim).to(args.device), requires_grad=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.ones = torch.tensor(1).to(args.device)
        self.zeros = torch.tensor(0).to(args.device)

        self.h = None
        self.batch_size = None
        self.ques_num = 2163
        with open('data/junyi/problem_skills_relation.pkl', 'rb') as f:
            ques_concept_mapping = pickle.load(f)
            ques_concept_table = torch.tensor(list(ques_concept_mapping.values()), device=self.device)
            zero_concept = torch.tensor([[0]], device=self.device)
            self.ques_concept_table = torch.cat([ques_concept_table, zero_concept], dim=0)

    def get_ques_representation(self, prob_ids, related_concept_index, data_len):
        filter0 = torch.where(related_concept_index == 0, self.zeros,  self.ones).float()
        concepts_cat = torch.cat(
            [torch.zeros(1, self.node_dim).to(self.device),
            self.concept_emb],
            dim = 0).unsqueeze(0).repeat(data_len, 1, 1)
        self.show_index = torch.arange(0, data_len).to(self.device)
        r_index = self.show_index.unsqueeze(1).repeat(1, self.max_concept)
        related_concepts = concepts_cat[r_index, related_concept_index,:]
        filter_sum = torch.sum(filter0, dim = 1)

        div = torch.where(filter_sum == 0, 
            torch.tensor(1.0).to(self.device), 
            filter_sum
            ).unsqueeze(1).repeat(1, self.node_dim)
        
        concept_level_rep = torch.sum(related_concepts, dim = 1) / div
        
        prob_cat = torch.cat([
            torch.zeros(1, self.node_dim).to(self.device),
            self.prob_emb], dim = 0)
        
        item_emb = prob_cat[prob_ids]

        v = torch.cat(
            [concept_level_rep,
            item_emb],
            dim = 1)
        return v

    def pi_cog_func(self, x, softmax_dim = 1):
        return F.softmax(self.select_preemb(x), dim = softmax_dim)

    def obtain_v(self, this_input, h, x, emb):
        
        prob_ids, related_concept_index, interval_time, concept_interval_time, elapsed_time, operate = this_input
        
        data_len = prob_ids.size()[0]
        filter0 = torch.where(related_concept_index == 0, self.ones, self.zeros).float()
        v = self.get_ques_representation(prob_ids, related_concept_index,  data_len)
        predict_x = torch.cat([h, v], dim = 1)
        h_v = torch.cat([h, v], dim = 1)
        prob = self.predictor(torch.cat([
            predict_x, emb
        ], dim = 1))
        return h_v, v, prob, x

    def update_state(self, h, v, emb, operate):

        v_cat = torch.cat([
            v.mul(operate.repeat(1, self.node_dim * 2)),
            v.mul((1 - operate).repeat(1, self.node_dim * 2))], dim = 1)
        e_cat = torch.cat([
            emb.mul((1-operate).repeat(1, self.node_dim * 2)),
            emb.mul((operate).repeat(1, self.node_dim * 2))], dim = 1)
        inputs = v_cat + e_cat
        next_p_state = self.gru_h(inputs, h)
        return next_p_state
    
    def pi_sens_func(self, x, softmax_dim = 1):
        return F.softmax(self.checker_emb(x), dim = softmax_dim)

    def forward(self, x):
        data_len = len(x[0])
        h = torch.zeros(data_len, self.node_dim).to(self.device)
        p_action_list, pre_state_list, emb_action_list, states_list, reward_list, predict_list, ground_truth_list = [], [], [], [], [], [], []

        rt_x = torch.zeros(data_len, 1, self.node_dim * 2).to(self.device)
        pre_hiddenstates = []
        for seqi in range(0, self.seq_length):
            ques_h = torch.cat([
                self.get_ques_representation(x[1][seqi][0], x[1][seqi][1], x[1][seqi][0].size()[0]),
                h], dim = 1)
            flip_prob_emb = self.pi_cog_func(ques_h)

            m = Categorical(flip_prob_emb)
            emb_ap = m.sample()
            emb_p = self.cog_matrix[emb_ap,:]

            h_v, v, logits, rt_x = self.obtain_v(x[1][seqi], h, rt_x, emb_p)
            prob = self.sigmoid(logits)
            out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)) 
            operate = out_operate_logits.float()
            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1-out_operate_logits).repeat(1, h_v.size()[-1]).float())],
                dim = 1)                
            out_x = torch.cat([out_x_logits, out_x_logits], dim = 1)

            ground_truth = x[1][seqi][5].squeeze(-1)

            flip_prob_emb = self.pi_sens_func(out_x)

            m = Categorical(flip_prob_emb)
            emb_a = m.sample()
            emb = self.acq_matrix[emb_a,:]

            h = self.update_state(h, v, emb, operate)
            
            emb_action_list.append(emb_a)
            p_action_list.append(emb_ap)
            states_list.append(out_x)
            pre_state_list.append(ques_h)
            
           
            predict_list.append(logits.squeeze(1))
            this_reward = torch.where(out_operate_logits.squeeze(1).float() == ground_truth,
                            torch.tensor(1).to(self.device), 
                            torch.tensor(0).to(self.device))
            reward_list.append(this_reward)

        seq_num = x[0]
        emb_action_tensor = torch.stack(emb_action_list, dim = 1)
        p_action_tensor = torch.stack(p_action_list, dim = 1)
        state_tensor = torch.stack(states_list, dim = 1)
        pre_state_tensor = torch.stack(pre_state_list, dim = 1)
        reward_tensor = torch.stack(reward_list, dim = 1).float() / (seq_num.unsqueeze(-1).repeat(1, self.seq_length)).float()
        logits_tensor = torch.stack(predict_list, dim = 1)
        
        loss, tracat_logits = [], []
        
        for i in range(0, data_len):
            this_seq_len = seq_num[i]
            this_reward_list = reward_tensor[i]
        
            this_cog_state = torch.cat([pre_state_tensor[i][0: this_seq_len],
                                    torch.zeros(1, pre_state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim = 0)
            this_sens_state = torch.cat([state_tensor[i][0: this_seq_len],
                                    torch.zeros(1, state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim = 0)

            td_target_cog = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_cog = td_target_cog
            delta_cog = delta_cog.detach().cpu().numpy()

            td_target_sens = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_sens = td_target_sens
            delta_sens = delta_sens.detach().cpu().numpy()

            advantage_lst_cog = []
            advantage = 0.0
            for delta_t in delta_cog[::-1]:
                advantage = self.gamma * advantage + delta_t[0]
                advantage_lst_cog.append([advantage])
            advantage_lst_cog.reverse()
            advantage_cog = torch.tensor(advantage_lst_cog, dtype=torch.float).to(self.device)
            
            pi_cog = self.pi_cog_func(this_cog_state[:-1])
            pi_a_cog = pi_cog.gather(1,p_action_tensor[i][0: this_seq_len].unsqueeze(1))

            loss_cog = -torch.log(pi_a_cog) * advantage_cog
            
            loss.append(torch.sum(loss_cog))

            advantage_lst_sens = []
            advantage = 0.0
            for delta_t in delta_sens[::-1]:
                advantage = self.gamma * advantage + delta_t[0]
                advantage_lst_sens.append([advantage])
            advantage_lst_sens.reverse()
            advantage_sens = torch.tensor(advantage_lst_sens, dtype=torch.float).to(self.device)
            
            pi_sens = self.pi_sens_func(this_sens_state[:-1])
            pi_a_sens = pi_sens.gather(1,emb_action_tensor[i][0: this_seq_len].unsqueeze(1))

            loss_sens = - torch.log(pi_a_sens) * advantage_sens
            loss.append(torch.sum(loss_sens))
            

            this_prob = logits_tensor[i][0: this_seq_len]
            tracat_logits.append(this_prob)

        return torch.cat(tracat_logits, dim = 0), self.lamb * sum(loss) / torch.sum(seq_num)

    def reset(self, batch_size=1):
        self.h = torch.zeros(batch_size, self.node_dim).to(self.device)
        self.batch_size = batch_size

        return self.h, []

    def get_prob_and_v(self, h, emb, batch_size, prob_ids, related_concept_index):
        v = self.get_ques_representation(prob_ids, related_concept_index, batch_size)
        predict_x = torch.cat([h, v], dim=1)
        h_v = torch.cat([h, v], dim=1)
        prob = self.predictor(torch.cat([
            predict_x, emb
        ], dim=1))
        return h_v, v, prob

    def step(self, h, batch_problem_id):
        with torch.no_grad():
            batch_problem_id.to(self.device)
            batch_concepts = self.ques_concept_table[batch_problem_id]
            self.h = h if h is not None else self.h

            ques_h = torch.cat([
                self.get_ques_representation(batch_problem_id, batch_concepts, self.batch_size),
                self.h], dim=1)

            # batch_size*10
            flip_prob_emb = self.pi_cog_func(ques_h)

            m = Categorical(flip_prob_emb)
            emb_ap = m.sample()
            emb_p = self.cog_matrix[emb_ap, :]

            h_v, v, logits = self.get_prob_and_v(self.h, emb_p, self.batch_size, batch_problem_id, batch_concepts)
            # prob: batch_size*1矩
            prob = self.sigmoid(logits)
            out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(self.device),
                                             torch.tensor(0).to(self.device))
            operate = out_operate_logits.float()
            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1 - out_operate_logits).repeat(1, h_v.size()[-1]).float())],
                dim=1)
            out_x = torch.cat([out_x_logits, out_x_logits], dim=1)

            flip_prob_emb = self.pi_sens_func(out_x)

            m = Categorical(flip_prob_emb)
            emb_a = m.sample()
            emb = self.acq_matrix[emb_a, :]

            self.h = self.update_state(h, v, emb, operate)

            batch_observation = out_operate_logits.reshape(-1)


        return self.h, batch_observation, [], []

    def total_probability(self):
        test_h = self.h.repeat(self.ques_num, 1)

        batch_problem_id = torch.arange(1, self.ques_num+1).repeat(self.batch_size, 1).T.reshape(-1)
        batch_concepts = self.ques_concept_table[batch_problem_id]

        ques_h = torch.cat([
            self.get_ques_representation(batch_problem_id, batch_concepts, self.batch_size),
            test_h], dim=1)

        flip_prob_emb = self.pi_cog_func(ques_h)

        m = Categorical(flip_prob_emb)
        emb_ap = m.sample()
        emb_p = self.cog_matrix[emb_ap, :]

        h_v, v, logits = self.get_prob_and_v(
            test_h, self.batch_size*self.ques_num, batch_problem_id, batch_concepts, emb_p)
        prob = self.sigmoid(logits)

        batch_prob = torch.split(prob.reshape(-1), self.batch_size)
        batch_prob = torch.stack(batch_prob, dim=0)

        total_probability = batch_prob.T

        return total_probability

    def test_target_score(self, target):
        with torch.no_grad():
            target_num = target.size()[1]

            test_h = self.h.repeat_interleave(target_num, dim=0)

            batch_problem_id = target.reshape(-1)
            batch_concepts = self.ques_concept_table[batch_problem_id]

            test_batch_size = self.batch_size * target_num

            ques_h = torch.cat([
                self.get_ques_representation(batch_problem_id, batch_concepts, test_batch_size),
                test_h], dim=1)

            flip_prob_emb = self.pi_cog_func(ques_h)

            m = Categorical(flip_prob_emb)
            emb_ap = m.sample()
            emb_p = self.cog_matrix[emb_ap, :]

            h_v, v, logits = self.get_prob_and_v(
                test_h, emb_p, test_batch_size, batch_problem_id, batch_concepts)
            prob = self.sigmoid(logits)

            batch_prob = torch.split(prob.reshape(-1), target_num)

            batch_prob = torch.stack(batch_prob, dim=0)

            batch_score = torch.sum(batch_prob, dim=1)

        return batch_score

    def test_massive_target_score(self, massive_target, test_times=10):
        with torch.no_grad():
            total_target_num = massive_target.size()[1]
            total_batch_score = torch.zeros(self.batch_size).to(self.device)

            for i in tqdm(range(test_times), 'testing score'):
                target_num = int(total_target_num / test_times)
                target = massive_target[:, i * target_num:(i + 1) * target_num]

                test_h = self.h.repeat_interleave(target_num, dim=0)

                batch_problem_id = target.reshape(-1)
                batch_concepts = self.ques_concept_table[batch_problem_id]

                test_batch_size = self.batch_size * target_num

                ques_h = torch.cat([
                    self.get_ques_representation(batch_problem_id, batch_concepts, test_batch_size),
                    test_h], dim=1)

                flip_prob_emb = self.pi_cog_func(ques_h)

                m = Categorical(flip_prob_emb)
                emb_ap = m.sample()
                emb_p = self.cog_matrix[emb_ap, :]

                h_v, v, logits = self.get_prob_and_v(
                    test_h, emb_p, test_batch_size, batch_problem_id, batch_concepts)
                prob = self.sigmoid(logits)

                batch_prob = torch.split(prob.reshape(-1), target_num)

                batch_prob = torch.stack(batch_prob, dim=0)

                batch_score = torch.sum(batch_prob, dim=1)

                total_batch_score += batch_score

        return total_batch_score
