'''有无用特征的'''
import tqdm
import os
import pickle
import logging as log
import torch
from torch.utils import data
# from torch_geometric.data import Data, Batch
import math
import random

class Dataset(data.Dataset):
    def __init__(self, problem_number, concept_num, root_dir, split='train'):
        super().__init__()
        self.map_dim = 0
        self.prob_encode_dim = 0
        self.path = root_dir
        self.problem_number = problem_number
        self.concept_num = concept_num
        self.show_len = 100
        self.split = split
        self.data_list = []
        # self.train_sample = train_sample
        log.info('Processing data...')
        self.process()
        log.info('Processing data done!')
        # self.count_0 = 0
        # self.total_pb = 0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def collate(self, batch):  # batch：从dataset中随机取出的200个元素组成的列表
        seq_num, y  = [], [] #seq_num: the actual length of history record 
        x = []
        seq_length = len(batch[0][1][1]) # the unifrom length of hitory record
        x_len = len(batch[0][1][0][0])
        # x_len = 9
        for i in range(0, seq_length):
            this_x = []
            for j in range(0, x_len):
                this_x.append([])
            x.append(this_x)
        for data in batch:
            this_seq_num, [this_x, this_y] = data
            seq_num.append(this_seq_num)
            for i in range(0, seq_length):
                # print(i, 'iiiiiiiiiiiiiiiiiiii')
                for j in range(0, x_len):
                    # print('iiiiiiiiiiiiiiii', i, j)
                    # this_x：200长列表，当前学生200步学习记录，每个元素是6长度列表，代表该学生6项信息（5个数一个列表）
                    # x: 200长列表，该batch学生200步学习记录，每个元素是6长度列表，代表该batch学生6项信息（6个列表，每个列表长batch_size）
                    x[i][j].append(this_x[i][j])
                # y[i].append(this_y[i])
            # y += this_y[1 : this_seq_num]
            y += this_y[0 : this_seq_num]
            # y += this_y
        batch_x, batch_y =[], []
        for i in range(0, seq_length):
            x_info = []
            for j in range(0, x_len):
                
                # if j == 2 or j == 6:
                if j != 5:    
                    x_info.append(torch.tensor(x[i][j]))
                else:
                    x_info.append(torch.tensor(x[i][j]).float())
            # x_info.append(x[i][j])
            # batch_x: seq_len长的列表，每个元素是6长度列表，每个元素是batch_size长度向量（或batch_size*4矩）,代表该步各学生信息
            batch_x.append(x_info)
            # batch_y.append(torch.tensor(y[i]))
        # return [seq_num, batch_x], torch.tensor(y).float()
        return [torch.tensor(seq_num), batch_x], torch.tensor(y).float()

 
    def data_reader(self, stu_records):
        '''
        @params:
            stu_record: learning history of a user
        @returns:
            x: question_id, skills, interval_time_to_previous, concept_interval_time, elapsed_time, correctness 
            y: response
        '''
        x_list = []
        y_list = []
        concepts_interval_time_count = dict()
        '''interval time = 0, the interval time is much large'''

        for i in range(0, len(stu_records)):
            # this_ques_id, this_tags, interval_with_pre, elapsed_time, resp
            problem_id, skills, interval_time, elapsed_time, response= stu_records[i]
       
            operate = [1]
            if response == 0:
                operate = [0] #避免除0报错

            '''process the interval time'''
            for c_str in concepts_interval_time_count.keys():
                concepts_interval_time_count[c_str] += interval_time
            
            this_skill_str = ''
            for s in skills:
                this_skill_str += str(s) + '-'
            this_skill_str = this_skill_str[:-1]
            if not this_skill_str in concepts_interval_time_count.keys():
                concepts_interval_time_count[this_skill_str] = 0

            this_concept_interval = concepts_interval_time_count[this_skill_str]
            
            x_list.append([
                problem_id,
                skills,
                interval_time,
                this_concept_interval,
                elapsed_time,
                operate
            ])

            y_list.append(torch.tensor(response))

        return x_list, y_list

    def process(self):
        self.prob_encode_dim = int(math.log(self.problem_number,2)) + 1
        with open(self.path + 'history_' + self.split + '.pkl', 'rb') as fp:
            histories = pickle.load(fp)
        loader_len = len(histories.keys())
        log.info('loader length: {:d}'.format(loader_len))
        proc_count = 0
        for k in tqdm.tqdm(histories.keys()):
            # if len(self.data_list) >= self.train_sample and self.split == 'train':
            #     break
            # if self.split == 'train' and proc_count > 3500:
            #     break
            # elif self.split == 'valid' and proc_count > 1200:
            #     break
            # if loader_len > 5000 and random.random() > 0.15:
            #     continue
            stu_record = histories[k]
            if stu_record[0] < 10:  # stu_record[0] 该学生的记录条数
                continue
            # print('length: ', stu_record[0])
            dt = self.data_reader(stu_record[1])  # stu_record[1] 该学生的回答记录 [200条记录, 200条记录的回答情况]
            
            # self.data_list.append([stu_record[0]] + dt)
            self.data_list.append([stu_record[0], dt])
            proc_count += 1
        log.info('final length {:d}'.format(len(self.data_list)))


