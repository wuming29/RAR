from git_envs.KSS.KSS_env_framework import BatchKSSFramework
# from git_envs.dkt_junyi.batch_DKT_junyi_framework import BatchDKTjunyiSimulator
# from git_envs.iekt_assist09.batch_iekt_assist09_framework import BatchIEKTassist09Simulator
# from git_envs.iekt_junyi.batch_iekt_junyi_framework import BatchIEKTjunyiSimulator
# from git_envs.dkt_assist09.batch_DKT_assist09_framework import BatchDKTassist09Simulator
from ORIAgent import ORIAgent
import sys
import torch
import random
import numpy as np
import argparse

# sys.path.append('git_envs/dkt_junyi')
# sys.path.append('git_envs/iekt_junyi')
# sys.path.append('git_envs/iekt_assist09')
# sys.path.append('git_envs/dkt_assist09')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default='KSS', type=str, help="IEKT_junyi, IEKT_assist09, DKT_junyi, DKT_assist09 or KSS")
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
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--ORI_device", default='cuda:0', type=str)
    parser.add_argument("--police_optimizer_refresh", default=True, type=bool)

    # env
    parser.add_argument("--episodes", default=1000, type=int)
    parser.add_argument("--init_records_len", default=20, type=int)
    parser.add_argument("--epoch_num", default=10, type=int)
    parser.add_argument("--target_num", default=400, type=int)
    parser.add_argument("--env_device", default='cuda:0', type=str)

    parser.add_argument("--seed", default=0, type=int)

    # parser.add_argument("--test_mode", default=True, type=bool)

    args = parser.parse_args()
    args.steps = args.n_steps

    # if args.env == 'IEKT_junyi':
    #     args.ques_num = 2163
    #     env = BatchIEKTjunyiSimulator(args)
    #     args.policy_mlp_hidden1 = 256
    #     args.policy_mlp_hidden2 = 512
    #     args.kt_mlp_hidden1 = 64
    #     args.kt_mlp_hidden2 = 32
    # 
    # elif args.env == 'IEKT_assist09':
    #     args.ques_num = 15003
    #     env = BatchIEKTassist09Simulator(args)
    #     args.policy_mlp_hidden1 = 512
    #     args.policy_mlp_hidden2 = 2048
    #     args.kt_mlp_hidden1 = 64
    #     args.kt_mlp_hidden2 = 32
    # 
    # elif args.env == 'DKT_junyi':
    #     args.ques_num = 2163
    #     env = BatchDKTjunyiSimulator(args)
    #     args.policy_mlp_hidden1 = 256
    #     args.policy_mlp_hidden2 = 512
    #     args.kt_mlp_hidden1 = 64
    #     args.kt_mlp_hidden2 = 32
    # 
    # elif args.env == 'DKT_assist09':
    #     args.ques_num = 15003
    #     env = BatchDKTassist09Simulator(args)
    #     args.policy_mlp_hidden1 = 512
    #     args.policy_mlp_hidden2 = 2048
    #     args.kt_mlp_hidden1 = 64
    #     args.kt_mlp_hidden2 = 32
    # 
    # else:

    args.ques_num = 10
    args.target_num = 10
    env = BatchKSSFramework()
    args.policy_mlp_hidden1 = 64
    args.policy_mlp_hidden2 = 32
    args.kt_mlp_hidden1 = 64
    args.kt_mlp_hidden2 = 32
    args.bc_batch_size = 256
    args.n_steps = 30
    
    setup_seed(args.seed)
    print(str(args))

    agent = ORIAgent(args)
    agent.name = 'ORI_seed{}'.format(args.seed)

    # if args.env == 'KSS':
    env.batch_train(agent=agent)
    # else:
    #     agent, saved_model, max_reward = env.batch_train(batch_size=args.batch_size, agent=agent)
    #     print('max reward = ', max_reward)
    # 
    #     mean_return = env.batch_test(batch_size=args.batch_size, agent=agent)
    #     print('mean_return: ', mean_return)
    # 
    #     mean_return = env.batch_test(batch_size=args.batch_size, agent=saved_model)
    #     print('saved_model_mean_return: ', mean_return)
    # 
    #     torch.save(agent, 'save_model/{}/final_{}.pt'.format(args.env, agent.name))
    #     torch.save(agent, 'save_model/{}/best_{}.pt'.format(args.env, agent.name))


main()
