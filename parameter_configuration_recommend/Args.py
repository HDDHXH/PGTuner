import  argparse

args = argparse.ArgumentParser()

args.add_argument('--method', type=str, default='ddpg', help='method')
args.add_argument('--params', type=str, default='', help='Load existing parameters')
args.add_argument('--memory', type=str, default='', help='add replay memory')
args.add_argument('--noisy', action='store_true', help='use noisy linear layer')
args.add_argument('--batch_size', type=int, default=128, help='Training Batch Size')
args.add_argument('--epoches', type=int, default=1, help='Training Epoches')
args.add_argument('--test_epoches', type=int, default=2000, help='Testing Epoches')
args.add_argument('--pec_reward', type=float, default=10)#5  #rec的奖励和占的比例
args.add_argument('--pec_rec', type=float, default=5)#5  #rec的奖励和占的比例
args.add_argument('--pec_ct', type=float, default=0.001)#0.1  #ct的奖励占的比例
args.add_argument('--pec_qps', type=float, default=1)#1  #qps的奖励占的比例
args.add_argument('--pec_ct_dt', type=float, default=0.05)#0.01, 0.05, 0.1  #ct距离计算次数的奖励占的比例
args.add_argument('--lamb', type=float, default=0.01)#0.01, 0.05, 0.1  #ct距离计算次数目标占的比例
args.add_argument('--alr', type=float, default=0.00001)#0.0001, 0.00001
args.add_argument('--clr', type=float, default=0.0001)#0.0001
args.add_argument('--alphalr', type=float, default=0.00001)
args.add_argument('--tau', type=float, default=0.0001)#0.0001
args.add_argument('--n_states', type=int, default=12, help='n_states') #旧
args.add_argument('--n_actions', type=int, default=3, help='n_actions')
args.add_argument('--n_states_nsg', type=int, default=18, help='n_states') #旧
args.add_argument('--n_actions_nsg', type=int, default=6, help='n_actions')
args.add_argument('--sigma_decay_rate', type=float, default=0.98) #暂时保持不变
args.add_argument('--sigma', type=float, default=0.2)#0.1
args.add_argument('--sigma_value', type=float, default=0.2)#0.2
args.add_argument('--policy_noise_clip', type=float, default=0.2)#0.3
args.add_argument('--delay_time', type=int, default=4)#0.3
args.add_argument('--memory_size', type=int, default=100000, help='memory_size')
args.add_argument('--max_steps', type=int, default=500, help='max_steps')
args.add_argument('--nochange_steps', type=int, default=200, help='max_nochange_steps')
args.add_argument('--nochange_episodes', type=int, default=200, help='max_nochange_episodes')
args.add_argument('--actor_layer_sizes', type=str, default='[128, 256, 256, 64]')
args.add_argument('--critic_layer_sizes', type=str, default='[256, 256, 256, 64]')
args.add_argument('--seed', type=int, default=42)#42

args, unknown = args.parse_known_args()
print(args)
