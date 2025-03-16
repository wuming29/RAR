from RARmodel import RARmodel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

class RARAgent:
    def __init__(self, args):
        self.name = 'RARAgent'
        self.env = '_'
        self.batch_size = args.batch_size
        self.device = args.RAR_device
        self.ques_num = args.ques_num
        self.n_steps = args.n_steps
        policy_mlp_hidden_dim_list = [args.hidden_dim*2, args.policy_mlp_hidden1, args.policy_mlp_hidden2, args.ques_num]
        kt_mlp_hidden_dim_list = [args.hidden_dim*2, args.kt_mlp_hidden1, args.kt_mlp_hidden2, 1]
        self.model = RARmodel(args.batch_size, args.ques_num, args.emb_dim, args.hidden_dim, args.weigh_dim,
                              args.target_num,
                              policy_mlp_hidden_dim_list, kt_mlp_hidden_dim_list, args.use_kt, 1,
                              args.n_head, args.n_layers, args.n_ques, self.device, m=args.psi)
        self.policy_optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.policy_optimizer_refresh = args.police_optimizer_refresh

        self.use_kt = args.use_kt
        self.alpha = args.alpha
        self.beta = args.beta

        self.action_list = []
        self.action_prob_list = []
        self.kt_prob_list = []
        self.observation_list = []

        self.history_exercise = []
        self.target = []

        self.step = 0

    def begin_epoch(self):
        if self.policy_optimizer_refresh:
            self.policy_optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)

    def begin_episode(self, exercises_record, target):
        self.action_list = []
        self.action_prob_list = []
        self.kt_prob_list = []
        self.observation_list = []

        self.history_exercise = []
        self.target = []

        self.model.initialize(exercises_record, target)
        self.target = target

        self.step = 0

    def take_action(self):
        action, prob = self.model.take_action()
        self.action_list.append(action.tolist())
        self.action_prob_list.append(prob)

        return action.tolist()

    def step_refresh(self, ques_id, observation):
        self.observation_list.append(observation)

        kt_prob = self.model.get_kt_prob(ques_id)
        self.model.step_refresh(ques_id, observation)

        self.kt_prob_list.append(kt_prob)

        self.step += 1

    def episode_refresh(self, reward, **kwargs):
        policy_loss = 0
        kt_loss = 0
        for step in range(len(self.action_prob_list)):
            for batch_id in range(len(self.action_prob_list[step])):
                sampled_actions = self.action_list[step][batch_id]
                this_batch_reward = torch.tensor(reward[batch_id]).to(self.device)
                policy_loss -= this_batch_reward * torch.log(self.action_prob_list[step][batch_id][sampled_actions])

        if self.use_kt:
            observation = torch.tensor(self.observation_list, dtype=torch.float).to(self.device)
            action_kt_probs = torch.stack(self.kt_prob_list)
            bce_loss_fn = nn.BCELoss()
            kt_loss = bce_loss_fn(action_kt_probs, observation)

        ranking_loss = self.model.get_ranking_loss(torch.stack(self.action_prob_list, dim=0))

        loss = policy_loss + self.alpha*kt_loss + self.beta*ranking_loss

        self.policy_optimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        self.policy_optimizer.step()

    def epoch_refresh(self):
        return

    def test_step_refresh(self, ques_id, observation):
        self.step += 1
        self.observation_list.append(observation)

    def test_episode_refresh(self):
        pass

