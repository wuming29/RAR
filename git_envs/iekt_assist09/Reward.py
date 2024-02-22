# coding:utf-8
# created by tongshiwei on 2018/11/13
import numpy as np


def get_reward(dataset, agent_kind):
    # return HybridReward()
    return GreedyExpReward()


class Reward(object):
    def __call__(self, initial_score, final_score, full_score, path, terminal_tag=False, rs=None):
        path_len = len(path)

        delta_base = full_score - initial_score
        delta = final_score - initial_score

        bias = 0

        def global_reward_decay(_global_reward):
            _global_reward = 0
            return _global_reward

        global_reward = delta
        normalize_factor = delta_base

        start = path_len - 1
        reward_values = [0] * path_len
        if terminal_tag:
            reward = global_reward
            reward_values[-1] = reward / normalize_factor
            start -= 1

        # bias
        defualt_r = bias

        rs = [defualt_r] * path_len if rs is None else rs

        assert len(rs) == path_len

        for i in range(start, -1, -1):
            # weight
            reward = global_reward + rs[i]
            global_reward = global_reward_decay(global_reward)
            reward_values[i] = reward / normalize_factor

        return reward_values

    @staticmethod
    def mature_reward(reward_values):
        reward_values = np.array(reward_values)
        eps = np.finfo(reward_values.dtype).eps.item()
        reward_values = (reward_values - reward_values.mean()) / (reward_values.std() + eps)
        return reward_values


class ExpReward(Reward):
    def __call__(self, initial_score, final_score, full_score, path, terminal_tag=False, rs=None):
        path_len = len(path)

        delta_base = full_score - initial_score
        delta = final_score - initial_score

        bias = 0

        def global_reward_decay(_global_reward):
            _global_reward *= (1 - 0.1 / delta_base)
            return _global_reward

        global_reward = delta
        normalize_factor = 1

        start = path_len - 1
        reward_values = [0] * path_len
        if terminal_tag:
            reward = global_reward
            reward_values[-1] = reward / normalize_factor
            start -= 1

        # bias
        defualt_r = bias

        rs = [defualt_r] * path_len if rs is None else rs

        assert len(rs) == path_len

        for i in range(start, -1, -1):
            # weight
            reward = global_reward + rs[i]
            global_reward = global_reward_decay(global_reward)
            reward_values[i] = reward / normalize_factor

        return reward_values


class GreedyExpReward(Reward):
    def __call__(self, initial_score, final_score, full_score, path, terminal_tag=False, rs=None):
        path_len = len(path)

        delta_base = full_score - initial_score
        delta = final_score - initial_score

        bias = 0

        def global_reward_decay(_global_reward):
            _global_reward *= 0.99
            return _global_reward

        global_reward = delta   # global_reward：最终得分 - 初始得分
        normalize_factor = full_score  # normalize_factor：总分，用于做分母

        start = path_len - 1  # 从后向前求回报
        reward_values = [0] * path_len
        if terminal_tag:  # terminal_tag表示是否推荐完了一个episode内应推荐的步数，还是半途终止了（动作空间最后一项是终止）
            reward = global_reward
            reward_values[-1] = reward / normalize_factor
            start -= 1

        # bias
        defualt_r = bias

        rs = [defualt_r] * path_len if rs is None else rs  # 不对reward加偏置

        assert len(rs) == path_len

        for i in range(start, -1, -1):
            # weight
            reward = global_reward + rs[i]
            global_reward = global_reward_decay(global_reward)  # 最终分数增长作为最后一步的回报，每往前一步，其回报都会有一个衰减（0.99）
            reward_values[i] = reward / normalize_factor

        return reward_values


class LinearReward(Reward):
    def __call__(self, initial_score, final_score, full_score, path, terminal_tag=False, rs=None):
        path_len = len(path)

        delta_base = full_score - initial_score
        delta = final_score - initial_score

        bias = 0

        def global_reward_decay(_global_reward):
            _global_reward -= 10
            return _global_reward

        global_reward = 1000 * delta - 500 * delta_base
        normalize_factor = delta_base

        start = path_len - 1
        reward_values = [0] * path_len
        if terminal_tag:
            reward = global_reward
            reward_values[-1] = reward / normalize_factor
            start -= 1

        # bias
        defualt_r = bias

        rs = [defualt_r] * path_len if rs is None else rs

        assert len(rs) == path_len

        for i in range(start, -1, -1):
            # weight
            reward = global_reward + rs[i]
            global_reward = global_reward_decay(global_reward)
            reward_values[i] = reward / normalize_factor

        return reward_values


class GreedyLinearReward(Reward):
    def __call__(self, initial_score, final_score, full_score, path, terminal_tag=False, rs=None):
        path_len = len(path)

        delta_base = full_score - initial_score
        delta = final_score - initial_score

        bias = 0

        def global_reward_decay(_global_reward):
            _global_reward -= 10
            return _global_reward

        global_reward = 1000 * delta - 500
        normalize_factor = delta_base

        start = path_len - 1
        reward_values = [0] * path_len
        if terminal_tag:
            reward = global_reward
            reward_values[-1] = reward / normalize_factor
            start -= 1

        # bias
        defualt_r = bias

        rs = [defualt_r] * path_len if rs is None else rs

        assert len(rs) == path_len

        for i in range(start, -1, -1):
            # weight
            reward = global_reward + rs[i]
            global_reward = global_reward_decay(global_reward)
            reward_values[i] = reward / normalize_factor

        return reward_values


class HybridReward(Reward):
    def __call__(self, initial_score, final_score, full_score, path, terminal_tag=False, rs=None):
        # 0.35 + 0.01 - 0.01 , 高出0.07
        # 0.73 + 0.01 - 0.01 , 高出0.13
        path_len = len(path)

        delta_base = full_score - initial_score
        delta = final_score - initial_score

        bias = 0

        level = initial_score / full_score

        def global_reward_decay(_global_reward):
            _global_reward *= 0.99
            return _global_reward

        if level >= 0.75:
            if 0.75 <= level < 0.85:
                global_reward = (delta - 0.5 * delta_base) / 0.5
            else:
                global_reward = (delta - 0.75 * delta_base) / 0.25

        elif 0.6 <= level < 0.75:
            global_reward = (delta - 0.25 * delta_base) / 0.75

        else:
            global_reward = (delta - 0.5) / (1 - 0.5 / delta_base)

        normalize_factor = delta_base

        start = path_len - 1
        reward_values = [0] * path_len
        if terminal_tag:
            reward = global_reward
            reward_values[-1] = reward / normalize_factor
            start -= 1

        # bias
        defualt_r = bias

        rs = [defualt_r] * path_len if rs is None else rs

        assert len(rs) == path_len

        for i in range(start, -1, -1):
            # weight
            reward = global_reward + rs[i]
            global_reward = global_reward_decay(global_reward)
            reward_values[i] = reward / normalize_factor

        return reward_values


class HybridReward2(Reward):
    def __call__(self, initial_score, final_score, full_score, path, terminal_tag=False, rs=None):
        # 0.35 + 0.01 -0.02 高出0.05
        # 0.66 + 0.01 -0.02 高出0.04
        path_len = len(path)

        delta_base = full_score - initial_score
        delta = final_score - initial_score

        bias = 0

        if initial_score / full_score >= 0.75:
            def global_reward_decay(_global_reward):
                _global_reward -= 10
                return _global_reward

            global_reward = 1000 * delta - 500 * delta_base

        elif 0.6 <= initial_score / full_score < 0.75:
            def global_reward_decay(_global_reward):
                _global_reward *= 0.99
                return _global_reward

            global_reward = 100 * delta - 50 * delta_base

        else:
            def global_reward_decay(_global_reward):
                _global_reward *= 0.99
                return _global_reward

            global_reward = 10 * delta

        normalize_factor = delta_base

        start = path_len - 1
        reward_values = [0] * path_len
        if terminal_tag:
            reward = global_reward
            reward_values[-1] = reward / normalize_factor
            start -= 1

        # bias
        defualt_r = bias

        rs = [defualt_r] * path_len if rs is None else rs

        assert len(rs) == path_len

        for i in range(start, -1, -1):
            # weight
            reward = global_reward + rs[i]
            global_reward = global_reward_decay(global_reward)
            reward_values[i] = reward / normalize_factor

        return reward_values
