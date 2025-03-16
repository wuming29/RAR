from git_envs.KSS.KSS_env_framework import BatchKSSFramework
from git_envs.dkt_junyi.batch_DKT_junyi_framework import BatchDKTjunyiSimulator
from git_envs.iekt_assist09.batch_iekt_assist09_framework import BatchIEKTassist09Simulator
from git_envs.iekt_junyi.batch_iekt_junyi_framework import BatchIEKTjunyiSimulator
from git_envs.dkt_assist09.batch_DKT_assist09_framework import BatchDKTassist09Simulator
from RARAgent import RARAgent
import sys
import torch
import random
import numpy as np
import argparse
import os
import time

sys.path.append('git_envs/dkt_junyi')
sys.path.append('git_envs/iekt_junyi')
sys.path.append('git_envs/iekt_assist09')
sys.path.append('git_envs/dkt_assist09')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default='DKTA09', type=str, help="IEKTJU, IEKTA09, DKTJU, DKTA09 or KSS")
    # agent
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--emb_dim", default=48, type=int, help="embedding dimension of model")
    parser.add_argument("--hidden_dim", default=64, type=int, help="hidden dimension of model")
    parser.add_argument("--weigh_dim", default=64, type=int, help="weight dimension of model")
    parser.add_argument("--policy_mlp_hidden1", default=256, type=int)
    parser.add_argument("--policy_mlp_hidden2", default=512, type=int)
    parser.add_argument("--kt_mlp_hidden1", default=64, type=int)
    parser.add_argument("--kt_mlp_hidden2", default=32, type=int)
    parser.add_argument("--use_kt", default=True, type=bool, help="whether use kt")
    parser.add_argument("--n_ques", default=1, type=int, help="recent_questions")
    parser.add_argument("--n_steps", default=200, type=int, help="rec_len")
    parser.add_argument("--n_head", default=1, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--paradigm_num", default=None, type=int)
    parser.add_argument("--alpha", default=1, type=int)
    parser.add_argument("--beta", default=None, type=float)
    parser.add_argument("--RAR_device", default='cuda:0', type=str)
    parser.add_argument("--police_optimizer_refresh", default=True, type=bool)
    parser.add_argument("--psi", default=None, type=float)

    # env
    parser.add_argument("--episodes", default=1000, type=int)
    parser.add_argument("--init_records_len", default=20, type=int)
    parser.add_argument("--epoch_num", default=10, type=int)
    parser.add_argument("--target_num", default=400, type=int)
    parser.add_argument("--env_device", default='cuda:0', type=str)

    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()
    args.steps = args.n_steps

    if args.env == 'IEKTJU' or args.env == 'IEKTjunyi':
        args.ques_num = 2163
        env = BatchIEKTjunyiSimulator(args)
        args.policy_mlp_hidden1 = 256
        args.policy_mlp_hidden2 = 512
        args.kt_mlp_hidden1 = 64
        args.kt_mlp_hidden2 = 32
        if args.psi == None:
            args.psi = 0.4
        if args.beta == None:
            args.beta = 0.1
    
    elif args.env == 'IEKTA09' or args.env == 'IEKTassist09':
        args.ques_num = 15003
        env = BatchIEKTassist09Simulator(args)
        args.policy_mlp_hidden1 = 512
        args.policy_mlp_hidden2 = 2048
        args.kt_mlp_hidden1 = 64
        args.kt_mlp_hidden2 = 32
        if args.psi == None:
            args.psi = 0.4
        if args.beta == None:
            args.beta = 0.1
    
    elif args.env == 'DKTJU' or args.env == 'DKTjunyi':
        args.ques_num = 2163
        env = BatchDKTjunyiSimulator(args)
        args.policy_mlp_hidden1 = 256
        args.policy_mlp_hidden2 = 512
        args.kt_mlp_hidden1 = 64
        args.kt_mlp_hidden2 = 32
        if args.psi == None:
            args.psi = 0.01
        if args.beta == None:
            args.beta = 0.01
    
    elif args.env == 'DKTA09' or args.env == 'DKTassist09':
        args.ques_num = 15003
        env = BatchDKTassist09Simulator(args)
        args.policy_mlp_hidden1 = 512
        args.policy_mlp_hidden2 = 2048
        args.kt_mlp_hidden1 = 64
        args.kt_mlp_hidden2 = 32
        if args.psi == None:
            args.psi = 0.6
        if args.beta == None:
            args.beta = 0.1
    
    elif args.env == 'KSS':
        args.ques_num = 10
        args.target_num = 10
        env = BatchKSSFramework()
        args.policy_mlp_hidden1 = 64
        args.policy_mlp_hidden2 = 32
        args.kt_mlp_hidden1 = 64
        args.kt_mlp_hidden2 = 32
        args.bc_batch_size = 256
        args.n_steps = 30
        if args.psi == None:
            args.psi = 15
        if args.beta == None:
            args.beta = 5

    else:
        print("Environment does not exist.")
        return
    
    setup_seed(args.seed)
    print(str(args))

    agent = RARAgent(args)
    ts = time.strftime('%H-%M-%S', time.localtime(time.time()))
    agent.name = 'RAR_seed{}_'.format(args.seed) + ts

    if not os.path.exists('save_model/{}'.format(args.env.replace('_', ""))):
        os.makedirs("save_model/{}".format(args.env.replace('_', "")))

    if args.env == 'KSS':
        env.batch_train(agent=agent)
    else:
        agent, saved_agent, max_reward = env.batch_train(batch_size=args.batch_size, agent=agent)
        print('max reward = ', max_reward)
    
        final_return = env.batch_test(batch_size=args.batch_size, agent=agent)
        print('mean_return: ', final_return)
    
        saved_return = env.batch_test(batch_size=args.batch_size, agent=saved_agent)
        print('saved_model_mean_return: ', saved_return)

        if final_return >= saved_return:
            torch.save(agent, 'save_model/{}/{}.pt'.format(args.env, agent.name))
        else:
            torch.save(saved_agent, 'save_model/{}/{}.pt'.format(args.env, agent.name))


main()
