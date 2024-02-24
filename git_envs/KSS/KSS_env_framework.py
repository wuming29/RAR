import logging

from . import KSS_env

from tqdm import tqdm

import matplotlib.pyplot as plt
import pickle
import torch


class KSSFramework:
    def __init__(self, max_steps=30, ctx=1, soft_discount=0, allow_shortcut=False, score_type=False,
                 graph_disable=False, steps_per_epoch=7500, epoch_num=10):
        self.max_steps = max_steps
        self.ctx = ctx
        self.soft_discount = soft_discount
        self.allow_shortcut = allow_shortcut
        self.score_type = score_type
        self.graph_disable = graph_disable
        self.steps_per_epoch = steps_per_epoch
        self.epoch_num = epoch_num
        self.logger = logging

        self.env = KSS_env.IRTEnvironment(student_num=4000, interactive=True)
        self.agent = None

        self.reward = self.env.reward

    def step(self):
        action = self.agent.take_action()

        exercise, correct = self.env.step(exercise=action)
        self.agent.step_refresh(exercise, correct)

        return exercise

    def episode_loop(self, max_steps_per_episode, desc=''):
        path = []

        exercises_record, target = self.env.begin_episode()
        self.agent.initialize(exercises_record, target)
        terminal_tag = False
        initial_score = self.env.test_score(target)

        for _ in tqdm(range(max_steps_per_episode), desc=desc):

            try:
                exercise = self.step()
                path.append(exercise)

            except StopIteration:
                terminal_tag = True
                break

        final_score = self.env.test_score(target)
        full_score = len(target)

        reward = final_score - initial_score
        normalize_factor = full_score - initial_score

        _episode_reward = reward / normalize_factor

        reward_values = self.reward(
            initial_score=initial_score,
            final_score=final_score,
            full_score=full_score,
            path=self.agent.path,
            terminal_tag=terminal_tag,
        )

        self.agent.episode_refresh(_episode_reward, reward_values)
        self.env.end_episode()

        return len(path), _episode_reward

    def epoch_loop(self, epoch, steps_per_epoch, max_steps_per_episode, desc=''):
        self.env.begin_epoch()

        steps_cnt = 0
        episode = 0
        _epoch_q_value = 0
        _epoch_reward = 0

        while steps_cnt <= steps_per_epoch:
            _ave_epoch_reward = _epoch_reward / episode if episode > 0 else float('nan')

            description = desc + "episode %d, steps: %d | %d, ave_reward %s" % (
                episode, steps_cnt, steps_per_epoch, _ave_epoch_reward)

            steps, reward = self.episode_loop(max_steps_per_episode, desc=description)
            steps_cnt += steps
            episode += 1

            _epoch_reward += reward

        self.env.end_epoch()
        self.agent.epoch_refresh()

        _ave_epoch_reward = _epoch_reward / episode if episode > 0 else float('nan')

        return _ave_epoch_reward

    def train(self, agent):
        self.agent = agent
        epoch_reward_list = []
        for e in range(self.epoch_num):
            desc = "epoch - %s | %s  " % (e, self.epoch_num)
            _ave_epoch_reward = self.epoch_loop(e, self.steps_per_epoch, self.max_steps, desc=desc)
            epoch_reward_list.append(_ave_epoch_reward)

        episode_list = [i+1 for i in range(self.epoch_num)]
        plt.plot(episode_list, epoch_reward_list)
        plt.xlabel('Epoch')
        plt.ylabel('Average Return')
        # plt.savefig('experiment_record/plot/{} on KSS.png'.format(self.agent.name))

        with open('experiment_record/train_record/KSS/{}.pkl'.format(self.agent.name), 'wb') as f:
            pickle.dump(epoch_reward_list, f)

        try:
            torch.save(self.agent, 'experiment_record/trained_agent_model/KSS/{}.pt'.format(self.agent.name))
        except TypeError:
            pass


        return self.agent, epoch_reward_list

    def test(self, agent, test_times=100):
        self.agent = agent
        ability_list = []  # test_times(100)个元素，每个元素是一个self.max_steps(30)个元素的列表，其中每个元素记录每一时间步时学生的能力（一个10个元素的列表）
        sum_ability_list = []  # test_times(100)个元素，每个元素是一个self.max_steps(30)个元素的列表，其中每个元素记录每一时间步时学生的能力总和（一个数）
        sum_ability = 0
        return_list = []
        for i in range(test_times):

            desc = "test {}, last ability:{}".format(i, sum_ability)
            episode_ability_list = []  # self.max_steps(30)个元素的列表，其中每个元素记录每一时间步时学生的能力（一个10个元素的列表）
            episode_sum_ability_list = []  # self.max_steps(30)个元素的列表，其中每个元素记录每一时间步时学生的能力（一个数）

            path = []
            exercises_record, target = self.env.begin_episode()
            self.agent.initialize(exercises_record, target)
            terminal_tag = False
            initial_score = self.env.test_score(target)

            for _ in tqdm(range(self.max_steps), desc=desc):

                try:
                    action = self.agent.take_action()
                    exercise, correct = self.env.step(exercise=action)
                    self.agent.step_refresh_test(exercise, correct)

                    ability = self.env._state
                    episode_ability_list.append(ability)
                    sum_ability = sum(ability)
                    episode_sum_ability_list.append(sum_ability)

                    path.append(exercise)

                except StopIteration:
                    terminal_tag = True
                    break

            final_score = self.env.test_score(target)
            full_score = len(target)

            reward = final_score - initial_score
            normalize_factor = full_score - initial_score

            _episode_reward = reward / normalize_factor

            ability_list.append(episode_ability_list)
            sum_ability_list.append(episode_sum_ability_list)
            return_list.append(_episode_reward)

            self.agent.episode_refresh_test()
            self.env.end_episode()

        with open('experiment_record/test_record/KSS/ability_list/{}.pkl'.format(self.agent.name), 'wb') as f:
            pickle.dump(ability_list, f)
        with open('experiment_record/test_record/KSS/return_list/{}.pkl'.format(self.agent.name), 'wb') as f:
            pickle.dump(sum_ability_list, f)

        mean_return = sum(return_list)/len(return_list)

        return sum_ability_list, return_list, mean_return

    def test_for_rltutor(self, agent):
        self.agent = agent

        self.env.begin_epoch()
        path = []
        exercises_record, target = self.env.begin_episode()
        self.agent.initialize(exercises_record)
        terminal_tag = False
        initial_score = self.env.test_score(target)

        for _ in tqdm(range(self.max_steps)):

            try:
                exercise = self.step()
                path.append(exercise)

            except StopIteration:
                terminal_tag = True
                break

        final_score = self.env.test_score(target)
        full_score = len(target)

        reward = final_score - initial_score
        normalize_factor = full_score - initial_score

        reward = reward / normalize_factor

        self.env.end_episode()

        print(path, 'reward: ', reward)

        return self.agent

    def test_for_generator(self, agent, max_steps):
        self.agent = agent

        self.env.begin_epoch()

        exercises_record, target = self.env.begin_episode()
        self.agent.initialize(exercises_record)
        initial_score = self.env.test_score(target)

        for _ in range(max_steps):

            try:
                exercise = self.step()

            except StopIteration:
                terminal_tag = True
                break

        final_score = self.env.test_score(target)
        full_score = len(target)

        reward = final_score - initial_score
        normalize_factor = full_score - initial_score

        reward = reward / normalize_factor

        self.env.end_episode()

        sa_record = self.agent.sa_record

        return reward, sa_record


class BatchKSSFramework:
    def __init__(self, batch_size=256, student_num_per_env=4000, max_steps=30, ctx=1, soft_discount=0,
                 allow_shortcut=False, score_type=False, graph_disable=False, steps_per_epoch=7500, epoch_num=10):
        self.batch_size = batch_size
        self.student_num_per_env = student_num_per_env
        self.max_steps = max_steps
        self.ctx = ctx
        self.soft_discount = soft_discount
        self.allow_shortcut = allow_shortcut
        self.score_type = score_type
        self.graph_disable = graph_disable
        self.steps_per_epoch = steps_per_epoch
        self.epoch_num = epoch_num
        self.logger = logging
        self.envs = []
        self.test_envs = []

        self.max_reward = 0

        self.valid_count = 0
        self.step_count = 0

        self.test_valid_count = 0
        self.test_step_count = 0

        pbar = tqdm(range(self.batch_size))
        for _ in pbar:
            pbar.set_description('generating data')
            env = KSS_env.IRTEnvironment(student_num=self.student_num_per_env, interactive=True)
            self.envs.append(env)

        pbar = tqdm(range(self.batch_size))
        for _ in pbar:
            pbar.set_description('generating test data')
            env = KSS_env.IRTEnvironment(student_num=self.student_num_per_env, interactive=True)
            self.test_envs.append(env)

        self.agent = None
        self.cs_agent = None

        self.rewards = [env.reward for env in self.envs]

    def step(self):
        action = self.agent.take_action()

        exercise_list = []
        correct_list = []
        valid_count = 0
        step_count = 0
        for idx, env in enumerate(self.envs):
            (exercise, correct), v = env.step(exercise=action[idx])
            exercise_list.append(exercise)
            correct_list.append(correct)
            valid_count += v
            step_count += 1

        self.agent.step_refresh(exercise_list, correct_list)

        return exercise_list, valid_count, step_count

    def episode_loop(self, max_steps_per_episode, desc=''):
        batch_path = []  # step_num(30) * batch_size的二维列表

        # 一、环境、agent初始化
        batch_exercises_record = []  # batch_size个元素, 每个元素是学生做题历史记录
        batch_target = []  # batch_size个元素，每个元素是target集合
        batch_initial_score = []  # batch_size个元素，每个元素是一个数

        for env in self.envs:
            exercises_record, target = env.begin_episode()
            initial_score = env.test_score(target)

            batch_exercises_record.append(exercises_record)
            batch_target.append(target)
            batch_initial_score.append(initial_score)

        self.agent.begin_episode(batch_exercises_record, batch_target)

        terminal_tag = [False for _ in range(len(self.envs))]

        # 二、推荐step_num（30）步
        for _ in tqdm(range(max_steps_per_episode), desc=desc):

            try:
                exercise_list, valid_count, step_count = self.step()
                self.valid_count += valid_count
                self.step_count += step_count
                batch_path.append(exercise_list)

            except StopIteration:
                terminal_tag = True
                break

        # 三、求奖励
        batch_final_score = []  # batch_size个元素，每个元素是一个数
        batch_full_score = []  # batch_size个元素，每个元素是一个数
        batch_reward = []  # batch_size个元素，每个元素是一个数
        batch_reward_values = []  # batch_size个元素，每个元素是一个列表（每一步的奖励）

        for idx, env in enumerate(self.envs):
            final_score = env.test_score(batch_target[idx])
            full_score = len(batch_target[idx])
            initial_score = batch_initial_score[idx]

            progress = final_score - initial_score
            normalize_factor = full_score - batch_initial_score[idx]
            reward = progress / normalize_factor

            #
            # reward_values = self.rewards[idx](
            #     initial_score=initial_score,
            #     final_score=final_score,
            #     full_score=full_score,
            #     path=self.agent.path,
            #     terminal_tag=terminal_tag,
            # )

            batch_final_score.append(final_score)
            batch_full_score.append(full_score)
            batch_reward.append(reward)
            # batch_reward_values.append(reward_values)

        # 四、更新环境和agent
        self.agent.episode_refresh(batch_reward, init_score=torch.tensor(batch_initial_score),
                                   aft_score=torch.tensor(batch_final_score),
                                   full_score=batch_full_score[0], terminal_tag=False)

        for env in self.envs:
            env.end_episode()

        return len(batch_path), batch_reward

    def epoch_loop(self, epoch, steps_per_epoch, max_steps_per_episode, desc=''):
        for env in self.envs:
            env.begin_epoch()

        steps_cnt = 0
        episode = 0
        _epoch_q_value = 0
        batch_epoch_reward = [0] * self.batch_size  # batch长的列表，每个元素记录该env下学生的每个episode下奖励之和

        self.agent.begin_epoch()

        self.valid_count = 0
        self.step_count = 0

        while steps_cnt <= steps_per_epoch:
            if episode > 0:
                batch_ave_epoch_reward = [epoch_reward / episode for epoch_reward in batch_epoch_reward]
                _ave_epoch_reward = str(sum(batch_ave_epoch_reward) / len(batch_ave_epoch_reward))
                hit_rate = str(self.valid_count / self.step_count)
            else:
                _ave_epoch_reward = 'nan'
                hit_rate = 'nan'

            description = desc + "episode %d, steps: %d | %d, ave_reward %s, hit_rate %s" % (
                episode, steps_cnt, steps_per_epoch, _ave_epoch_reward, hit_rate)

            steps, batch_reward = self.episode_loop(max_steps_per_episode, desc=description)
            steps_cnt += steps
            episode += 1

            batch_epoch_reward = [i + j for i, j in zip(batch_epoch_reward, batch_reward)]

        for env in self.envs:
            env.end_epoch()
        self.agent.epoch_refresh()

        # 平均每个batch的平均每个episode的奖励
        _ave_epoch_reward = sum(batch_epoch_reward)/len(batch_epoch_reward) / episode if episode > 0 else float('nan')

        return _ave_epoch_reward

    def batch_train(self, agent):
        self.agent = agent
        epoch_reward_list = []
        for e in range(self.epoch_num):
            desc = "epoch - %s | %s  " % (e, self.epoch_num)
            _ave_epoch_reward = self.epoch_loop(e, self.steps_per_epoch, self.max_steps, desc=desc)
            epoch_reward_list.append(_ave_epoch_reward)

            return_list, mean_return = self.batch_test(test_times=100)

            if mean_return > self.max_reward:
                self.max_reward = mean_return
                if self.agent.name == 'CSEAL':
                    from mxnet import ndarray
                    net = self.agent.agent.value_net.net_mod.net
                    params = net._collect_params_with_prefix()
                    arg_dict = {key: val._reduce() for key, val in params.items()}
                    ndarray.save('save_model/KSS/{}.parmas'.format(self.agent.name), arg_dict)
                else:
                    torch.save(self.agent, 'save_model/KSS/{}.pt'.format(self.agent.name))

        episode_list = [i+1 for i in range(self.epoch_num)]
        plt.plot(episode_list, epoch_reward_list)
        plt.xlabel('Epoch')
        plt.ylabel('Average Return')
        # plt.savefig('experiment_record/plot/{} on KSS.png'.format(self.agent.name))

        with open('experiment_record/train_record/KSS/{}.pkl'.format(self.agent.name), 'wb') as f:
            pickle.dump(epoch_reward_list, f)

        try:
            torch.save(self.agent, 'experiment_record/trained_agent_model/KSS/{}.pt'.format(self.agent.name))
        except TypeError:
            pass

        if self.agent.name == 'CSEAL':
            save_model = None
        else:
            save_model = torch.load('save_model/KSS/{}.pt'.format(self.agent.name), map_location=self.agent.device)

        return self.agent, save_model, self.max_reward

    def batch_test(self, test_times=100):
        # self.agent = agent
        ability_list = []  # test_times(100)个元素，每个元素是一个self.max_steps(30)个元素的列表，其中每个元素记录每一时间步时学生的能力（一个10个元素的列表）
        sum_ability_list = []  # test_times(100)个元素，每个元素是一个self.max_steps(30)个元素的列表，其中每个元素记录每一时间步时学生的能力总和（一个数）
        sum_ability = 0
        return_list = []
        # self.envs = []

        self.test_valid_count = 0
        self.test_step_count = 0

        # pbar = tqdm(range(self.batch_size))
        # for _ in pbar:
        #     pbar.set_description('generating test data')
        #     env = KSS_env.IRTEnvironment(student_num=self.student_num_per_env, interactive=True)
        #     self.envs.append(env)

        for i in range(test_times):

            desc = "test {}, last ability:{}".format(i, sum_ability)
            episode_ability_list = []  # self.max_steps(30)个元素的列表，其中每个元素记录每一时间步时学生的能力（一个10个元素的列表）
            episode_sum_ability_list = []  # self.max_steps(30)个元素的列表，其中每个元素记录每一时间步时学生的能力（一个数）

            path = []

            # 一、环境、agent初始化
            batch_exercises_record = []  # batch_size个元素, 每个元素是学生做题历史记录
            batch_target = []  # batch_size个元素，每个元素是target集合
            batch_initial_score = []  # batch_size个元素，每个元素是一个数

            for env in self.test_envs:
                exercises_record, target = env.begin_episode()
                initial_score = env.test_score(target)

                batch_exercises_record.append(exercises_record)
                batch_target.append(target)
                batch_initial_score.append(initial_score)

            self.agent.begin_episode(batch_exercises_record, batch_target)
            terminal_tag = [False for _ in range(len(self.envs))]

            for _ in tqdm(range(self.max_steps), desc=desc):

                try:
                    action = self.agent.take_action()

                    exercise_list = []
                    correct_list = []
                    for idx, env in enumerate(self.test_envs):
                        (exercise, correct), v = env.step(exercise=action[idx])
                        exercise_list.append(exercise)
                        correct_list.append(correct)
                        self.test_valid_count += v
                        self.test_step_count += 1

                    self.agent.step_refresh(exercise_list, correct_list)

                    # ability = self.env._state
                    # episode_ability_list.append(ability)
                    # sum_ability = sum(ability)
                    # episode_sum_ability_list.append(sum_ability)

                    path.append(exercise_list)

                except StopIteration:
                    terminal_tag = True
                    break

            batch_final_score = []  # batch_size个元素，每个元素是一个数
            batch_full_score = []  # batch_size个元素，每个元素是一个数
            batch_reward = []  # batch_size个元素，每个元素是一个数
            batch_reward_values = []  # batch_size个元素，每个元素是一个列表（每一步的奖励）

            for idx, env in enumerate(self.test_envs):
                final_score = env.test_score(batch_target[idx])
                full_score = len(batch_target[idx])
                initial_score = batch_initial_score[idx]

                progress = final_score - initial_score
                normalize_factor = full_score - batch_initial_score[idx]
                reward = progress / normalize_factor

                batch_final_score.append(final_score)
                batch_full_score.append(full_score)
                batch_reward.append(reward)

            return_list.append(sum(batch_reward)/len(batch_reward))

            self.agent.test_episode_refresh()

            for env in self.envs:
                env.end_episode()

        with open('experiment_record/test_record/KSS/return_list/{}.pkl'.format(self.agent.name), 'wb') as f:
            pickle.dump(return_list, f)

        mean_return = sum(return_list)/len(return_list)

        print('hit_rate: ', self.test_valid_count/self.test_step_count)
        print('mean_return: ', mean_return)

        return return_list, mean_return
