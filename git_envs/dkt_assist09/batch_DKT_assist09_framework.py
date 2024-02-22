import torch
import random
from tqdm import tqdm
import numpy as np
import os
import inspect
import matplotlib.pyplot as plt
import pickle
from .Reward import GreedyExpReward


class BatchDKTassist09Simulator:
    def __init__(self, args):
        self.agent = None

        self.steps = args.steps
        self.episodes = args.episodes
        self.init_records_len = args.init_records_len
        self.device = torch.device(args.env_device)
        self.h = None
        self.epoch_num = args.epoch_num
        self.ques_num = args.ques_num + 1
        self.ques_list = [i for i in range(self.ques_num)]

        for frame_info in inspect.stack():
            if 'batch_DKT_assist09_framework.py' in frame_info.filename:
                self.path = os.path.dirname(frame_info.filename)
        self.env = torch.load(os.path.abspath(self.path+'/dkt_assist09_simulator.pt'),
                              map_location=self.device)
        self.env.device = self.device
        self.reward = GreedyExpReward()
        self.target_num = args.target_num

    def batch_train(self, batch_size, agent):
        self.agent = agent
        max_reward = -1

        for epoch_id in range(self.epoch_num):
            self.agent.begin_epoch()
            epoch_reward_list = []
            with tqdm(total=int(self.episodes / self.epoch_num), desc='Iteration %d' % epoch_id) as pbar:
                for i_episode in range(int(self.episodes / self.epoch_num)):

                    batch_target = []  # batch_size个元素，每个元素代表当前env的target
                    # 一、初始化每一个环境
                    self.h, _ = self.env.reset(batch_size)
                    # 生成历史答题记录，交给agent初始化
                    # 1. 初始化学生模型
                    exercises_record = []
                    agent_exercises_record = []

                    # batch_size个学生的历史学习记录
                    for step_id in range(self.init_records_len):
                        batch_init_ques_id = torch.randint(1, self.ques_num, (batch_size,)).to(self.device)
                        self.h, batch_observation, _, _ = self.env.step(self.h, batch_init_ques_id)
                        exercises_record.append(torch.stack([batch_init_ques_id, batch_observation], dim=1))
                        agent_exercises_record.append(torch.stack([batch_init_ques_id - 1, batch_observation], dim=1))
                    # exercises_record: self.init_records_len个元素的列表，每个元素是batch_size长的[ques_id, correct]对列表
                    # 问题下标从1开始
                    batch_init_exercises_record = torch.stack(exercises_record, dim=1)  # batch_size*self.init_records_len*2
                    # 问题下标从0开始
                    agent_batch_init_exercises_record = torch.stack(agent_exercises_record, dim=1).tolist()  # batch_size*self.init_records_len*2
                    # 2. 初始化学习目标
                    target_prob = torch.randn(batch_size, self.ques_num-1).to(self.device)

                    # 以下两个均为batch_size * self.target_num矩
                    target = torch.topk(target_prob, k=self.target_num)[1] + 1
                    batch_target = (target-1).tolist()
                    index = torch.arange(0, batch_size).repeat(self.target_num, 1).T

                    batch_target_table = torch.full((batch_size, self.ques_num), False).to(self.device)
                    batch_target_table[index, target] = torch.full((batch_size, self.target_num), True).to(self.device)

                    # 3. 获取初始分数
                    batch_score_init = self.env.test_target_score(target)

                    self.agent.begin_episode(agent_batch_init_exercises_record, batch_target)

                    # 二、每个环境都做steps步交互
                    for step in range(self.steps):
                        ques_id_list = self.agent.take_action()
                        batch_ques_id = torch.tensor(ques_id_list, device=self.device) + 1
                        self.h, batch_observation, _, _ = self.env.step(self.h, batch_ques_id)

                        observation_list = batch_observation.tolist()

                        self.agent.step_refresh(ques_id_list, observation_list)

                    # 三、归一化的进步值作为奖励
                    batch_score_aft = self.env.test_target_score(target)
                    # all_score = self.env.total_probability
                    # # batch_size * ques_num, 是学习目标的位置标记学生得分，否则置为0
                    # batch_target_score_table = torch.where(batch_target_table, all_score, 0)
                    # # batch_size长的向量，标记每个学生的初始得分
                    # batch_score_aft = torch.sum(batch_target_score_table, dim=1)
                    # batch_size长的向量，记录每个学生的学习效果
                    batch_reward = (batch_score_aft - batch_score_init) / (self.target_num - batch_score_init)
                    epoch_reward_list.append(torch.mean(batch_reward).item())

                    self.agent.episode_refresh(batch_reward, init_score=batch_score_init, aft_score=batch_score_aft,
                                               full_score=self.target_num, terminal_tag=False)

                    # 更新进度条
                    pbar.set_postfix({
                        'episode':
                            '%d' % (self.episodes / 10 * epoch_id + i_episode + 1),
                        'ave_score_after':
                            '%.6f' % torch.mean(batch_reward)
                    })
                    pbar.update(1)

                    this_reward = torch.mean(batch_reward)
                    if this_reward > max_reward:
                        max_reward = this_reward
                        if self.agent.name == 'CSEAL':
                            from mxnet import ndarray
                            net = self.agent.agent.value_net.net_mod.net
                            params = net._collect_params_with_prefix()
                            arg_dict = {key: val._reduce() for key, val in params.items()}
                            ndarray.save('save_model/DKTassist09/{}.parmas'.format(self.agent.name), arg_dict)
                        else:
                            torch.save(self.agent, 'save_model/DKTassist09/{}.pt'.format(self.agent.name))

            self.agent.epoch_refresh()
            print(epoch_reward_list)
        if self.agent.name == 'CSEAL':
            save_model = None
        else:
            save_model = torch.load('save_model/DKTassist09/{}.pt'.format(self.agent.name), map_location=self.agent.device)

        return self.agent, save_model, max_reward

    def batch_test(self, batch_size, agent, test_times=100):
        self.agent = agent

        self.agent.begin_epoch()

        batch_ave_reward_list = []

        with tqdm(total=test_times) as pbar:
            for i_episode in range(test_times):
                batch_target = []  # batch_size个元素，每个元素代表当前env的target
                # 一、初始化每一个环境
                self.h, _ = self.env.reset(batch_size)
                # 生成历史答题记录，交给agent初始化
                # 1. 初始化学生模型
                exercises_record = []
                agent_exercises_record = []

                # batch_size个学生的历史学习记录
                for step_id in range(self.init_records_len):
                    batch_init_ques_id = torch.randint(1, self.ques_num, (batch_size,)).to(self.device)
                    self.h, batch_observation, _, _ = self.env.step(self.h, batch_init_ques_id)
                    exercises_record.append(torch.stack([batch_init_ques_id, batch_observation], dim=1))
                    agent_exercises_record.append(torch.stack([batch_init_ques_id - 1, batch_observation], dim=1))
                # exercises_record: self.init_records_len个元素的列表，每个元素是batch_size长的[ques_id, correct]对列表
                batch_init_exercises_record = torch.stack(exercises_record, dim=1)  # batch_size*self.init_records_len*2
                agent_batch_init_exercises_record = torch.stack(agent_exercises_record, dim=1)

                # 2. 初始化学习目标
                target_prob = torch.randn(batch_size, self.ques_num-1).to(self.device)

                # 以下两个均为batch_size * self.target_num矩
                target = torch.topk(target_prob, k=self.target_num)[1] + 1
                batch_target = (target - 1).tolist()
                index = torch.arange(0, batch_size).repeat(self.target_num, 1).T

                batch_target_table = torch.full((batch_size, self.ques_num), False).to(self.device)
                batch_target_table[index, target] = torch.full((batch_size, self.target_num), True).to(self.device)

                # 3. 获取初始分数
                batch_score_init = self.env.test_target_score(target)

                self.agent.begin_episode(agent_batch_init_exercises_record.tolist(), batch_target)

                # 二、每个环境都做steps步交互
                for step in range(self.steps):
                    ques_id_list = self.agent.take_action()
                    batch_ques_id = torch.tensor(ques_id_list, device=self.device) + 1
                    self.h, batch_observation, _, _ = self.env.step(self.h, batch_ques_id)
                    observation_list = batch_observation.tolist()
                    self.agent.test_step_refresh(ques_id_list, observation_list)

                # 三、归一化的进步值作为奖励
                batch_score_aft = self.env.test_target_score(target)

                # all_score = self.env.total_probability
                # # batch_size * ques_num, 是学习目标的位置标记学生得分，否则置为0
                # batch_target_score_table = torch.where(batch_target_table, all_score, 0)
                # # batch_size长的向量，标记每个学生的初始得分
                # batch_score_aft = torch.sum(batch_target_score_table, dim=1)
                # batch_size长的向量，记录每个学生的学习效果
                batch_reward = (batch_score_aft - batch_score_init) / (self.target_num - batch_score_init)

                batch_ave_reward = torch.mean(batch_reward)
                batch_ave_reward_list.append(batch_ave_reward.item())

                self.agent.test_episode_refresh()

                # 更新进度条
                pbar.set_postfix({
                    'episode':
                        '%d' % (i_episode + 1),
                    'ave_score_after':
                        '%.6f' % torch.mean(batch_reward).item()
                })
                pbar.update(1)

        test_mean_reward = sum(batch_ave_reward_list) / len(batch_ave_reward_list)

        return test_mean_reward

    def train(self, agent):
        self.agent = agent
        return_list = []

        for i in range(self.epoch_num):
            with tqdm(total=int(self.episodes / self.epoch_num), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.episodes / self.epoch_num)):
                    self.h, total_probability = self.env.reset()

                    # 生成历史答题记录，交给agent初始化
                    exercises_record = []
                    for _ in range(self.init_records_len):
                        ques_id = random.randint(0, 2163)
                        self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
                        exercises_record.append([ques_id, observation])

                    target_num = random.randint(300, 500)
                    target = set(random.sample(self.ques_list, target_num))

                    all_score = self.env.total_probability[0]
                    target_score_init = 0
                    for target_id in target:
                        target_score_init += all_score[target_id].item()

                    self.agent.initialize(exercises_record, target)

                    # agent为模拟学生推荐steps道题目
                    for step in range(self.steps):
                        try:
                            ques_id = self.agent.take_action()
                        except StopIteration:
                            break

                        self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
                        self.agent.step_refresh(ques_id, observation)

                    all_score = self.env.total_probability[0]
                    target_score_aft = 0
                    for target_id in target:
                        target_score_aft += all_score[target_id].item()

                    return_value = (target_score_aft - target_score_init)/(len(target) - target_score_init)
                    return_list.append(return_value)  # 学习完200个项目后的得分列表，episodes(1000)个元素
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode':
                                '%d' % (self.episodes / 10 * i + i_episode + 1),
                            'return':
                                '%.3f' % np.mean(return_list[-10:])
                        })

                    reward_values = self.reward(
                        initial_score=target_score_init,
                        final_score=target_score_aft,
                        full_score=len(target),
                        path=self.agent.path,
                        terminal_tag=False,
                    )

                    self.agent.episode_refresh(return_value, reward_values)

                    pbar.update(1)

                self.agent.epoch_refresh()

        episode_list = [i for i in range(self.episodes)]
        plt.plot(episode_list, return_list)
        plt.xlabel('episode')
        plt.ylabel('final_score')
        plt.savefig('experiment_record/plot/{} on DKT_assist09.png'.format(self.agent.name))

        with open('experiment_record/train_record/DKT_assist09/episode_return_list/{}.pkl'.format(self.agent.name), 'wb') as f:
            pickle.dump(return_list, f)

        try:
            torch.save(self.agent, 'experiment_record/trained_agent_model/DKT_assist09/{}.pt'.format(self.agent.name))
        except TypeError:
            pass

        return self.agent, return_list

    def test(self, agent, test_times=100):
        self.agent = agent
        score_list = []
        growth_list = []
        return_list = []

        with tqdm(total=test_times) as pbar:
            for i_episode in range(int(test_times)):
                self.h, total_probability = self.env.reset()

                # 生成历史答题记录，交给agent初始化
                exercises_record = []
                for _ in range(self.init_records_len):
                    ques_id = random.randint(0, 2163)
                    self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
                    exercises_record.append([ques_id, observation])

                target_num = random.randint(300, 500)
                target = set(random.sample(self.ques_list, target_num))

                all_score = self.env.total_probability[0]
                target_score_init = 0
                for target_id in target:
                    target_score_init += all_score[target_id].item()

                self.agent.initialize(exercises_record, target)

                # agent为模拟学生推荐steps道题目
                for step in range(self.steps):
                    ques_id = self.agent.take_action()
                    self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)

                    self.agent.step_refresh_test(ques_id, observation)

                all_score = self.env.total_probability[0]
                target_score_aft = 0
                for target_id in target:
                    target_score_aft += all_score[target_id].item()

                return_value = (target_score_aft - target_score_init) / (len(target) - target_score_init)
                return_list.append(return_value)  # 学习完200个项目后的得分列表，test_time(100)个元素
                pbar.set_postfix({
                    'episode':
                        '%d' % (i_episode + 1),
                    'return':
                        '%.3f' % return_list[-1]
                })

                self.agent.episode_refresh_test()

                pbar.update(1)

        mean_reward = np.mean(return_list)

        return mean_reward

    def train_for_rltutor(self, agent):
        self.agent = agent
        score_list = []

        self.h, total_probability = self.env.reset()

        # 生成历史答题记录，交给agent初始化
        exercises_record = []
        for _ in range(self.init_records_len):
            ques_id = random.randint(0, self.ques_num)
            self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
            exercises_record.append([ques_id, observation])
        self.agent.initialize(exercises_record)

        # agent为模拟学生推荐steps道题目
        for step in range(20):
            for _ in range(10):
                ques_id = self.agent.take_action()
                self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
                score_list.append(score_aft)
                self.agent.step(ques_id, observation.item())

            self.agent.save_and_refresh()

        return score_list

    def test_for_generator(self, agent, max_steps):
        self.agent = agent
        score_list = []
        score_aft = 0

        self.h, total_probability = self.env.reset()

        # 生成历史答题记录，交给agent初始化
        exercises_record = []
        for _ in range(self.init_records_len):
            ques_id = random.randint(0, self.ques_num)
            self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
            exercises_record.append([ques_id, observation])
        self.agent.initialize(exercises_record)

        # agent为模拟学生推荐steps道题目
        for step in range(max_steps):
            ques_id = self.agent.take_action()
            self.h, observation, reward, score_aft = self.env.step(self.h, ques_id)
            score_list.append(score_aft)
            self.agent.step(ques_id, observation.item())

        reward = score_aft
        sa_record = self.agent.sa_record

        return reward, sa_record
