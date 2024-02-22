# coding: utf-8
# create by tongshiwei on 2018/11/28
import math
import json
from tqdm import tqdm
from .longling import wf_open
from .Reward import GreedyExpReward
import random


from .Graph import graph_candidate

import networkx as nx

graph_edges = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (2, 8),
    (3, 4),
    (4, 8),
    (5, 4),
    (5, 9),
    (6, 7),
    (7, 8),
    (8, 9),
]

ORDER_RATIO = 1

# coding: utf-8
# create by tongshiwei on 2018/11/28
import json
import random
import logging

from .longling import clock
from .longling import flush_print
from .longling.lib.candylib import as_list
import torch


class Env(object):
    def __init__(self, *args, **kwargs):
        self.reward = None
        self._state = None
        self._initial_state = None
        self.score_for_test = None
        self.logger = logging
        self.interactive = kwargs.get('interactive', True)

    @property
    def mastery(self):
        raise NotImplementedError

    @property
    def state(self):
        return self.mastery

    def begin_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def end_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def begin_episode(self, *args, **kwargs):
        raise NotImplementedError

    def end_episode(self, *args, **kwargs):
        raise NotImplementedError

    def is_valid_sample(self, target):
        if self.test_score(target) == len(target):
            return False
        return True

    def remove_invalid_student(self, idx):
        raise NotImplementedError

    @property
    def student_num(self):
        raise NotImplementedError

    def get_student(self, idx):
        raise NotImplementedError

    def step(self, exercise, **kwargs):
        # get函数以第一个参数查找键，没有该键则返回第二个参数。step中传入的是空字典，而self.interactive = True，因此走这边
        if kwargs.get('interactive', self.interactive) is True:
            # self(exercise)：传入多个exercise时返回[(exercise, correct), ...]，否则只返回当前exercise的[(exercise, correct)]
            return self(exercise)[0][0]
        else:  # 不走这边
            return self(exercise)[0][0][0], None

    def step4myagent(self, exercise, **kwargs):
        # get函数以第一个参数查找键，没有该键则返回第二个参数。step中传入的是空字典，而self.interactive = True，因此走这边
        if kwargs.get('interactive', self.interactive) is True:
            # self(exercise)：传入多个exercise时返回[(exercise, correct), ...]，否则只返回当前exercise的[(exercise, correct)]
            out = self(exercise)
            prob = torch.tensor([self.mastery])
            skill = torch.tensor([self._state])
            return out[0][0], out[1], prob, skill
        else:  # 不走这边
            return self(exercise)[0][0][0], self(exercise)[0][1], None

    def __call__(self, exercises):
        assert self._state
        exercises = as_list(exercises)

        exercises_record = []
        for exercise in exercises:
            correct = self.correct(exercise)  # 往下第二个函数就是，根据答对概率随机生成是否答对的标签
            exercises_record.append((exercise, correct))
            p_mastery = sum(self._state)
            self.state_transform(exercise, correct)
            a_mastety = sum(self._state)
            reward = a_mastety - p_mastery - 0.25
        return exercises_record, reward

    def state_transform(self, exercise, correct=None):
        raise NotImplementedError

    def correct(self, exercise):
        return 1 if random.random() <= self.mastery[exercise] else 0

    def test_correct(self, exercise, mastery):
        return 1 if 0.5 <= mastery[exercise] else 0

    def test(self, exercises, score_type=False, mastery=None):  # score_type=False初始未传入
        mastery = mastery if mastery is not None else self.mastery
        if score_type:
            return [(exercise, mastery[exercise]) for exercise in exercises]
        return [(exercise, self.test_correct(exercise, mastery)) for exercise in exercises]

    def test_score(self, exercises, score_type=None, mastery=None):
        score_type = self.score_for_test if score_type is None else score_type
        return sum([s for _, s in self.test(exercises, score_type, mastery)])



def irt(ability, difficulty, c=0.25):
    discrimination = 5
    return c + (1 - c) / (1 + math.exp(-1.7 * discrimination * (ability - difficulty)))


class IRTEnvironment(Env):
    def __init__(self, student_num=4000, seed=10, **kwargs):
        super(IRTEnvironment, self).__init__(**kwargs)
        self.path = None

        random.seed(seed)

        self.reward = GreedyExpReward()
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(graph_edges)
        self.topo_order = list(nx.topological_sort(self.graph))
        self.default_order = [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]
        self.difficulty = self.get_ku_difficulty(len(self.graph.nodes), self.topo_order)

        self.students = self.generate_students(student_num)

        self._target = None
        self._legal_candidates = None

        random.seed(None)

    @property
    def mastery(self):  # 这个方法返回答对概率，而非能力值
        # self._state是某一位学生的对各题目的能力值（一维、10元素列表存储）
        return [irt(_state, self.difficulty[idx]) for idx, _state in enumerate(self._state)]

    def dump_id2idx(self, filename):
        with open(filename, "w") as wf:
            for node in self.graph.nodes:
                print("%s,%s" % (node, node), file=wf)

    def dump_graph_edges(self, filename):
        with open(filename, "w") as wf:
            for edge in self.graph.edges:
                print("%s,%s" % edge, file=wf)

    def generate_students(self, student_num, step=20):
        student_abilities = self.get_student_ability(student_num)  # [学生1能力[题目1能力, 题目2能力, ...], 学生2能力, ...]
        students = []

        # for student_ability in tqdm(student_abilities, "loading data"):
        for student_ability in student_abilities:
            self._state = student_ability[:]
            exercises_record = []  # 将会记录[(exercise,correct), ...]
            cnt = 0
            if random.random() < ORDER_RATIO:  # ORDER_RATIO = 1
                while cnt < step:
                    cnt += 1
                    # 前一题答对并前三题没做同样的题目
                    if exercises_record and exercises_record[-1][1] == 1 and len(
                            set([e[0] for e in exercises_record[-3:]])) > 1:
                        for _ in range(1):
                            # 再做一遍上一道题，并记录
                            exercises_record.append(self.step(exercises_record[-1][0], interactive=True))
                        # 记录node，让他再做一遍
                        node = exercises_record[-1][0]
                    # 上题没答对：0.7概率再做一遍
                    elif exercises_record and exercises_record[-1][1] == 0 and random.random() < 0.7:
                        node = exercises_record[-1][0]
                    # 上题答对并作过相同题目，或上题没答对，0.3概率，或没有上题：0.9概率按拓扑排序选题（根据知识结构选题）
                    elif random.random() < 0.9:
                        for node in self.topo_order:
                            if self.mastery[node] < 0.6:
                                break
                        else:
                            break
                    # 0.1概率随机选
                    else:
                        node = random.randint(0, len(self.topo_order) - 1)
                    # self.step返回(exercise,correct)并更新学生状态： self.state_transform(exercise, correct)
                    exercises_record.append(self.step(node, interactive=True))
            else:
                while cnt < step:
                    cnt += 1
                    if random.random() < 0.9:
                        for node in self.default_order:
                            if self.mastery[node] < 0.6:
                                break
                        else:
                            break
                    else:
                        node = random.randint(0, len(self.topo_order) - 1)
                    exercises_record.append(self.step(node))

            # 这里是依照学生能力生成的学习历史记录，每生成一步，通过step方法（下的state_transform）更新学生能力（还是学生状态固定？）
            # exercises_record:[(exercise1, correct1), (exercise2, correct2), ...]
            students.append([student_ability, exercises_record,#  set([i for i in range(10)])])
                             set(random.sample(self.graph.nodes, random.randint(3, len(self.graph.nodes))))])
            # students:[学生1信息[学生能力，历史学习记录，学习目标], ...]
        return students

    def sim_seq(self, step):
        exercises_record = []
        cnt = 0
        if random.random() < ORDER_RATIO:
            while cnt < step:
                cnt += 1
                if exercises_record and exercises_record[-1][1] == 1 and len(
                        set([e[0] for e in exercises_record[-3:]])) > 1:
                    for _ in range(1):
                        exercises_record.append(self.step(exercises_record[-1][0]))
                    node = exercises_record[-1][0]
                elif exercises_record and exercises_record[-1][1] == 0 and random.random() < 0.7:
                    node = exercises_record[-1][0]
                elif random.random() < 0.9:
                    for node in self.topo_order:
                        if self.mastery[node] < 0.6:
                            break
                    else:
                        break
                else:
                    node = random.randint(0, len(self.topo_order) - 1)
                exercises_record.append(self.step(node))
        else:
            while cnt < step:
                cnt += 1
                if random.random() < 0.9:
                    for node in self.default_order:
                        if self.mastery[node] < 0.6:
                            break
                    else:
                        break
                else:
                    node = random.randint(0, len(self.topo_order) - 1)
                exercises_record.append(self.step(node))
        return exercises_record

    def dump_kt(self, student_num, filename, step=50):
        students = self.get_student_ability(student_num)

        with wf_open(filename) as wf:
            for student in tqdm(students, "simirt for kt"):
                self._state = student[:]
                exercises_record = []
                cnt = 0
                if random.random() < ORDER_RATIO:
                    while cnt < step:
                        cnt += 1
                        # 有历史记录，且上题答对，且前三题不是同一道
                        if exercises_record and exercises_record[-1][1] == 1 and len(
                                set([e[0] for e in exercises_record[-3:]])) > 1:
                            # 重做本题
                            for _ in range(1):
                                exercises_record.append(self.step(exercises_record[-1][0]))
                            node = exercises_record[-1][0]
                        # 有历史记录且上题答错，0.7概率重做
                        elif exercises_record and exercises_record[-1][1] == 0 and random.random() < 0.7:
                            node = exercises_record[-1][0]
                        # 上题答对且已经做过3次，或上题答错（0.3概率），0.9概率：找下一个掌握度不好的题目
                        elif random.random() < 0.9:
                            for node in self.topo_order:
                                if self.mastery[node] < 0.6:
                                    break
                            else:
                                break
                        else:
                            node = random.randint(0, len(self.topo_order) - 1)
                        exercises_record.append(self.step(node))
                else:
                    while cnt < step:
                        cnt += 1
                        if random.random() < 0.9:
                            for node in self.default_order:
                                if self.mastery[node] < 0.6:
                                    break
                            else:
                                break
                        else:
                            node = random.randint(0, len(self.topo_order) - 1)
                        exercises_record.append(self.step(node))
                print(json.dumps(exercises_record), file=wf)

    @property
    def student_num(self):
        return len(self.students)

    @staticmethod
    def get_student_ability(student_num):
        return [[random.randint(-3, 0) - (0.1 * i) for i in range(10)] for _ in range(student_num)]

    @staticmethod
    def get_ku_difficulty(ku_num, order):
        _difficulty = sorted([random.randint(0, 5) for _ in range(ku_num)])
        difficulty = [0] * ku_num
        for index, j in enumerate(order):
            difficulty[j] = _difficulty[index]
        return difficulty

    def state_transform(self, exercise, correct=None):
        graph = self.graph
        a = self._state
        ind = exercise

        if self.path:
            if exercise not in graph_candidate(graph, a, self.path[-1], allow_shortcut=False, target=self._target,
                                               legal_candidates=self._legal_candidates)[0]:
                return

        if self.path is not None:
            self.path.append(exercise)

        # predecessors是父节点（前驱节点），前面掌握的越好discount越小，ratio越大，increase越多，本题的答对概率增加越多
        discount = math.exp(sum([(5 - a[node]) for node in graph.predecessors(ind)] + [0]))
        ratio = 1 / discount
        inc = (5 - a[ind]) * ratio * 0.5

        def _promote(_ind, _inc):
            a[_ind] += _inc
            if a[_ind] > 5:
                a[_ind] = 5
            # 回答本题还会增加后续知识掌握情况，增量依次向后折半
            for node in graph.successors(_ind):
                _promote(node, _inc * 0.5)

        _promote(ind, inc)

    def begin_epoch(self):
        pass

    def end_epoch(self):
        pass

    def begin_episode(self, target_all=False):
        while True:
            _idx = random.randint(0, len(self.students) - 1)
            # 返回第_idx个学生的历史学习记录，以及target
            exercises, target = self.get_student(_idx)
            if target_all:
                target = set(range(10))
            if self.is_valid_sample(target):
                return exercises, target
            else:
                self.remove_invalid_student(_idx)

    def end_episode(self):
        self.path = None
        self._target = None
        self._legal_candidates = None
        self._state = None
        self._initial_state = None

    def get_student(self, idx):
        student = self.students[idx]
        target = student[2]
        self._state = student[0][:]
        self._initial_state = student[0][:]
        self.path = [student[1][-1][0]]
        self._target = set(target)
        self._legal_candidates = set(target)
        return student[1], target


if __name__ == '__main__':
    # for irt
    # env = IRTEnvironment(None, 0)
    # data_dir = "../data/irt/data/"
    # env.dump_kt(4000, data_dir + "dataset")
    # env.dump_graph_edges(data_dir + "graph_edges.idx")
    # env.dump_id2idx(data_dir + "vertex_id2idx")

    env = IRTEnvironment(None, 0)
    data_dir = "../data/kss/data/"
    # env.dump_kt(10000, data_dir + "dataset", step=200)
    # 这里生成的学生记录是至少50步的，这里的学生没有记录在self.students里，只是用于生成模拟数据。与环境交互时的学生是通过begin_episode方法，
    # 调用self.student属性（该属性值通过generate_students方法获得）得到的，他们有至少20步的学习记录，且返回的学生能力值是20步记录前的能力值，
    # 但记录生成时能力值存在更新没有更新
    env.dump_kt(4000, data_dir + "dataset", step=50)  #
    env.dump_graph_edges(data_dir + "graph_edges.idx")
    env.dump_id2idx(data_dir + "vertex_id2idx")

    # for simirt
    # env = IRTEnvironment(None, 0)
    # data_dir = "../data/simirt/data/"
    # env.dump_kt(8000, data_dir + "sim_dataset")
    # env.dump_graph_edges(data_dir + "graph_edges.idx")
    # env.dump_id2idx(data_dir + "vertex_id2idx")

    with open(data_dir + "sim_dataset") as f, wf_open(data_dir + "dataset") as wf, wf_open(
            data_dir + "rec_dataset.raw") as rec_wf:
        for line in f:
            if random.random() < 0.5:
                print(line, end='', file=wf)
            else:
                print(line, end='', file=rec_wf)
