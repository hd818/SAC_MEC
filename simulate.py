# -*- coding: utf-8 -*-

import pybullet_envs
import matplotlib.pyplot as plt

from constants import *
from environment import MECsystem
from decision import Agent
from decision import ReplayBuffer
import matplotlib.pyplot as plt
import gym
from gym import wrappers


UEnet = Agent(input_dims=[8], env=None, n_actions=2)
'''原设定参数
 UEnet = Agent(alpha=0.000025, beta=0.00025, input_dims = 8, tau=0.001, \
              env=None, batch_size=64, layer1_size=500, layer2_size=300,
              n_actions=1)
'''
env = MECsystem(apply_num, UEnet)
MECSnet = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])

'''原设定参数
MECSnet = Agent(alpha=0.000025, beta=0.00025, input_dims = \
              8*apply_num+BS2MECS_rate.size*channel_gain.size+1,
              tau=0.001, env=env, batch_size=64, layer1_size=500,
              layer2_size=300, n_actions=apply_num*4)
'''

np.random.seed(0)

n_games= 300     #episode次数修改在这里

list=[]

score_history = []
for i in range(n_games):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        act = MECSnet.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        MECSnet.remember(obs, act, reward, new_state, int(done))
        MECSnet.learn()
        score += reward
        obs = new_state
        print('reward is： {}'.format(reward))
        list.append(reward)
    score_history.append(score)
#
#     print('episode ', i, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))     ##取最后一百场game并取平均值以了解学习情况
#     if i % 25 == 0:                     ##每25场game保存模型
#         UEnet.save_models()
# filename = 'MEC_offloading.png'
# plot_learning(score_history, filename, 'window  =100')

print(list)
plt.plot(np.arange(len(list)),list)
plt.xlabel('time')
plt.ylabel('formatReward')

plt.grid()
plt.savefig('figure2.eps', format='eps', dpi=1000)
plt.show()

'''
plt.plot(format(reward))
plt.ylabel('Return')
plt.xlabel("Episode")
plt.grid(True)
plt.show()

'''
