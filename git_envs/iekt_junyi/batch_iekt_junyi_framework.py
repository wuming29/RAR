import torch
from tqdm import tqdm
import os
import inspect


class BatchIEKTjunyiSimulator:
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
            if 'batch_iekt_junyi_framework.py' in frame_info.filename:
                self.path = os.path.dirname(frame_info.filename)
        self.env = torch.load(os.path.abspath(self.path+'/iekt_junyi_simulator.pt'),
                              map_location=self.device)
        self.env.device = self.device
        self.target_num = args.target_num

    def batch_train(self, batch_size, agent):
        self.agent = agent
        max_reward = -1

        for epoch_id in range(self.epoch_num):
            self.agent.begin_epoch()
            epoch_reward_list = []
            with tqdm(total=int(self.episodes / self.epoch_num), desc='Iteration %d' % epoch_id) as pbar:
                for i_episode in range(int(self.episodes / self.epoch_num)):

                    self.h, _ = self.env.reset(batch_size)

                    exercises_record = []
                    agent_exercises_record = []

                    for step_id in range(self.init_records_len):
                        batch_init_ques_id = torch.randint(1, self.ques_num, (batch_size,)).to(self.device)
                        self.h, batch_observation, _, _ = self.env.step(self.h, batch_init_ques_id)
                        exercises_record.append(torch.stack([batch_init_ques_id, batch_observation], dim=1))
                        agent_exercises_record.append(torch.stack([batch_init_ques_id - 1, batch_observation], dim=1))

                    agent_batch_init_exercises_record = torch.stack(agent_exercises_record, dim=1).tolist()

                    target_prob = torch.randn(batch_size, self.ques_num-1).to(self.device)

                    target = torch.topk(target_prob, k=self.target_num)[1] + 1
                    batch_target = (target-1).tolist()
                    index = torch.arange(0, batch_size).repeat(self.target_num, 1).T

                    batch_target_table = torch.full((batch_size, self.ques_num), False).to(self.device)
                    batch_target_table[index, target] = torch.full((batch_size, self.target_num), True).to(self.device)

                    batch_score_init = self.env.test_target_score(target)

                    self.agent.begin_episode(agent_batch_init_exercises_record, batch_target)

                    for step in range(self.steps):
                        ques_id_list = self.agent.take_action()
                        batch_ques_id = torch.tensor(ques_id_list, device=self.device) + 1
                        self.h, batch_observation, _, _ = self.env.step(self.h, batch_ques_id)

                        observation_list = batch_observation.tolist()

                        self.agent.step_refresh(ques_id_list, observation_list)

                    batch_score_aft = self.env.test_target_score(target)

                    batch_reward = (batch_score_aft - batch_score_init) / (self.target_num - batch_score_init)
                    epoch_reward_list.append(torch.mean(batch_reward).item())

                    self.agent.episode_refresh(batch_reward, init_score=batch_score_init, aft_score=batch_score_aft,
                                               full_score=self.target_num, terminal_tag=False)

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
                            ndarray.save('save_model/IEKTJU/{}.parmas'.format(self.agent.name), arg_dict)

                        else:
                            torch.save(self.agent, 'save_model/IEKTJU/{}.pt'.format(self.agent.name))

            self.agent.epoch_refresh()

        if self.agent.name == 'CSEAL':
            saved_model = None

        else:
            saved_model = torch.load('save_model/IEKTJU/{}.pt'.format(self.agent.name), map_location=self.agent.device)

        return self.agent, saved_model, max_reward

    def batch_test(self, batch_size, agent, test_times=100):
        self.agent = agent

        self.agent.begin_epoch()

        batch_ave_reward_list = []

        with tqdm(total=test_times) as pbar:
            for i_episode in range(test_times):
                batch_target = []

                self.h, _ = self.env.reset(batch_size)

                exercises_record = []
                agent_exercises_record = []

                for step_id in range(self.init_records_len):
                    batch_init_ques_id = torch.randint(1, self.ques_num, (batch_size,)).to(self.device)
                    self.h, batch_observation, _, _ = self.env.step(self.h, batch_init_ques_id)
                    exercises_record.append(torch.stack([batch_init_ques_id, batch_observation], dim=1))
                    agent_exercises_record.append(torch.stack([batch_init_ques_id - 1, batch_observation], dim=1))

                agent_batch_init_exercises_record = torch.stack(agent_exercises_record, dim=1)

                target_prob = torch.randn(batch_size, self.ques_num-1).to(self.device)

                target = torch.topk(target_prob, k=self.target_num)[1] + 1
                batch_target = (target - 1).tolist()
                index = torch.arange(0, batch_size).repeat(self.target_num, 1).T

                batch_target_table = torch.full((batch_size, self.ques_num), False).to(self.device)
                batch_target_table[index, target] = torch.full((batch_size, self.target_num), True).to(self.device)

                batch_score_init = self.env.test_target_score(target)

                self.agent.begin_episode(agent_batch_init_exercises_record.tolist(), batch_target)

                for step in range(self.steps):
                    ques_id_list = self.agent.take_action()
                    batch_ques_id = torch.tensor(ques_id_list, device=self.device) + 1
                    self.h, batch_observation, _, _ = self.env.step(self.h, batch_ques_id)
                    observation_list = batch_observation.tolist()
                    self.agent.test_step_refresh(ques_id_list, observation_list)

                batch_score_aft = self.env.test_target_score(target)

                batch_reward = (batch_score_aft - batch_score_init) / (self.target_num - batch_score_init)

                batch_ave_reward = torch.mean(batch_reward)
                batch_ave_reward_list.append(batch_ave_reward.item())

                self.agent.test_episode_refresh()

                pbar.set_postfix({
                    'episode':
                        '%d' % (i_episode + 1),
                    'ave_score_after':
                        '%.6f' % torch.mean(batch_reward).item()
                })
                pbar.update(1)

        test_mean_reward = sum(batch_ave_reward_list) / len(batch_ave_reward_list)

        return test_mean_reward
