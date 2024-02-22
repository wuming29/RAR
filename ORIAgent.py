import random

from ORImodel import ORImodel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import pickle
from tqdm import trange

class ORIAgent:
    def __init__(self, args):
        self.name = 'ORIAgent'
        self.env = '_'
        self.batch_size = args.batch_size
        self.bc_every_epoch_episode = args.bc_every_epoch_episode
        self.bc_every_episode_episode = args.bc_every_episode_episode
        self.device = args.ORI_device
        self.ques_num = args.ques_num
        self.n_steps = args.n_steps
        policy_mlp_hidden_dim_list = [args.hidden_dim*2, args.policy_mlp_hidden1, args.policy_mlp_hidden2, args.ques_num]
        kt_mlp_hidden_dim_list = [args.hidden_dim*2, args.kt_mlp_hidden1, args.kt_mlp_hidden2, 1]
        self.model = ORImodel(args.batch_size, args.ques_num, args.emb_dim, args.hidden_dim, args.weigh_dim,
                              args.target_num,
                              policy_mlp_hidden_dim_list, kt_mlp_hidden_dim_list, args.use_kt, 1,
                              args.n_head, args.n_layers, args.n_ques, self.device)
        self.bc_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.policy_optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.bc_optimizer_refresh = args.bc_optimizer_refresh
        self.policy_optimizer_refresh = args.police_optimizer_refresh

        self.bc_paradigm_path = args.bc_paradigm_path
        self.bc_student_info_path = args.bc_student_info_path

        self.use_kt = args.use_kt
        self.bc_every_epoch = args.bc_every_epoch
        self.bc_every_episode = args.bc_every_episode
        self.alpha = args.alpha
        self.beta = args.beta

        print(self.beta)

        self.action_list = []
        self.action_prob_list = []
        self.kt_prob_list = []
        self.observation_list = []

        self.history_exercise = []
        self.target = []

        self.step = 0

        if args.init_bc:
            self.behavior_cloning(args.bc_begin_episode, batch_size=args.bc_batch_size, paradigm_num=args.paradigm_num)

        # with open(args.bc_student_info_path, 'rb') as f1:
        #     self.original_students_info = pickle.load(f1)
        #
        # with open(args.bc_paradigm_path, 'rb') as f2:
        #     self.paradigm = pickle.load(f2)

    def begin_epoch(self):
        if self.bc_optimizer_refresh:
            self.bc_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        if self.policy_optimizer_refresh:
            self.policy_optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        if self.bc_every_epoch:
            self.behavior_cloning(self.bc_every_epoch_episode)

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
        bc_loss = 0
        if self.bc_every_episode:
            bc_loss = self.get_bc_loss(self.bc_every_episode_episode)
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
        # loss = policy_loss + self.alpha * kt_loss

        self.policy_optimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        self.policy_optimizer.step()

    def epoch_refresh(self):
        return

    def behavior_cloning(self, n_episode, batch_size=None, paradigm_num=None):
        with open(self.bc_student_info_path, 'rb') as f1:
            original_students_info = pickle.load(f1)

        with open(self.bc_paradigm_path, 'rb') as f2:
            paradigm = pickle.load(f2)

        paradigm_id_list = [i for i in range(len(paradigm))]
        if paradigm_num:
            paradigm_id_list = random.sample(paradigm_id_list, paradigm_num)

        acc_list = []
        if not batch_size:
            batch_size = self.batch_size
        with trange(n_episode) as t:
            for _ in t:
                t.set_description("behavior cloning")

                kt_loss = torch.tensor(0, dtype=torch.float).to(self.device)

                batch_student_id = random.sample(paradigm_id_list, batch_size)
                batch_exercises_record = [original_students_info[student_id][1] for student_id in batch_student_id]
                batch_target = [original_students_info[student_id][2] for student_id in batch_student_id]

                batch_paradigm_action_list = [[paradigm[student_id][step][0] for student_id in batch_student_id]
                                              for step in range(len(paradigm[0]))]
                batch_paradigm_observation_list = [[paradigm[student_id][step][1] for student_id in batch_student_id]
                                                   for step in range(len(paradigm[0]))]

                action_list, action_prob_list, kt_prob_list = [], [], []
                self.model.initialize(batch_exercises_record, batch_target, batch_size)
                for step_id in range(self.n_steps):
                    action, prob = self.model.take_action()
                    action = action.tolist()
                    action_list.append(action)
                    action_prob_list.append(prob)

                    kt_prob = self.model.get_kt_prob(action)
                    kt_prob_list.append(kt_prob)

                    self.model.step_refresh(action, batch_paradigm_observation_list[step_id])

                action_onehot = F.one_hot(torch.tensor(batch_paradigm_action_list), num_classes=self.ques_num).to(self.device)

                action_prob = torch.stack(action_prob_list).to(self.device)

                bce_loss_fn = nn.BCELoss()
                bc_loss = bce_loss_fn(action_prob, action_onehot.float())

                if self.use_kt:
                    observation = torch.tensor(batch_paradigm_observation_list, dtype=torch.float).to(self.device)
                    action_kt_probs = torch.stack(kt_prob_list).to(self.device)
                    bce_loss_fn = nn.BCELoss()
                    kt_loss = bce_loss_fn(action_kt_probs, observation)

                loss = kt_loss + bc_loss
                self.bc_optimizer.zero_grad()
                loss.backward()
                self.bc_optimizer.step()

                correct = 0
                count = 0
                for step_id in range(len(batch_paradigm_action_list)):
                    for batch_id in range(len(batch_paradigm_action_list[step_id])):
                        if batch_paradigm_action_list[step_id][batch_id] == action_list[step_id][batch_id]:
                            correct += 1
                        count += 1

                acc = correct / count
                acc_list.append(acc)

                t.set_postfix({'loss': loss.item(), 'acc': acc})

        episode_list = [i for i in range(n_episode)]

    def test_step_refresh(self, ques_id, observation):
        self.step += 1
        self.observation_list.append(observation)

    def test_episode_refresh(self):
        pass

    def get_bc_loss(self, n_episode):

        with trange(n_episode) as t:
            for _ in t:
                t.set_description("behavior cloning")
                kt_loss = torch.tensor(0, dtype=torch.float).to(self.device)

                batch_student_id = random.sample([i for i in range(len(self.paradigm))], self.batch_size)
                batch_exercises_record = [self.original_students_info[student_id][1] for student_id in batch_student_id]
                batch_target = [self.original_students_info[student_id][2] for student_id in batch_student_id]

                batch_paradigm_action_list = [[self.paradigm[student_id][step][0] for student_id in batch_student_id]
                                              for step in range(len(self.paradigm[0]))]
                batch_paradigm_observation_list = [[self.paradigm[student_id][step][1]
                                                    for student_id in batch_student_id]
                                                   for step in range(len(self.paradigm[0]))]

                action_list, action_prob_list, kt_prob_list = self.model(batch_exercises_record, batch_target)

                action_onehot = F.one_hot(torch.tensor(batch_paradigm_action_list), num_classes=10).to(self.device)
                action_prob = torch.stack(action_prob_list).to(self.device)

                bce_loss_fn = nn.BCELoss()
                bc_loss = bce_loss_fn(action_prob, action_onehot.float())

                if self.use_kt:
                    observation = torch.tensor(batch_paradigm_observation_list, dtype=torch.float).to(self.device)
                    action_kt_probs = torch.stack(kt_prob_list).to(self.device)
                    bce_loss_fn = nn.BCELoss()
                    kt_loss = bce_loss_fn(action_kt_probs, observation)

                loss = kt_loss + bc_loss

            return loss
