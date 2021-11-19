# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt

from constants import *
from environment import MECsystem
from decision import Agent







UEnet = Agent( alpha=0.0003, beta=0.0003, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2)
'''原设定参数
 UEnet = Agent(alpha=0.000025, beta=0.00025, input_dims = 8, tau=0.001, \
              env=None, batch_size=64, layer1_size=500, layer2_size=300,
              n_actions=1)
'''
env = MECsystem(apply_num, UEnet)
MECSnet = Agent(alpha=0.0003, beta=0.0003, input_dims = \
    (8*apply_num+BS2MECS_rate.size*channel_gain.size+1,),
              tau=0.005, env=env, max_size=1000000, batch_size=256, layer1_size=256,
              layer2_size=256, n_actions=apply_num*8,reward_scale=2)
print('8*apply_num+BS2MECS_rate.size*channel_gain.size+1=',8*apply_num+BS2MECS_rate.size*channel_gain.size+1)

'''原设定参数
MECSnet = Agent(alpha=0.000025, beta=0.00025, input_dims = \
              8*apply_num+BS2MECS_rate.size*channel_gain.size+1,
              tau=0.001, env=env, batch_size=64, layer1_size=500,
              layer2_size=300, n_actions=apply_num*4)
'''

np.random.seed(0)

# n_games= 56    #episode次数修改在这里

list=[]


score_history = []
##原ddpgのsimulate
for i in range(10):
    done = False
    score = 0
    obs = env.reset()

    print('started')
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
#     print('episode ', i, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))     ##取最后一百场game并取平均值以了解学习情况
#     if i % 25 == 0:                     ##每25场game保存模型
#         UEnet.save_models()
# filename = 'MEC_offloading.png'
# plot_learning(score_history, filename, 'window  =100')


#####################################SAC改版simulate
# for i in range(n_games):
#     observation = env.reset()
#     done = False
#     score = 0
#     while not done:
#         action = MECSnet.choose_action(observation)
#         new_state, reward, done, info = env.step(action)
#         score += reward
#         MECSnet.remember(observation, action, reward, new_state, done)
#         MECSnet.learn()
#         observation=new_state
#     score_history.append(score)
#     avg_score = np.mean(score_history[-100:])
#
#     if avg_score > best_score:
#         best_score = avg_score
#
#     print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
#     list.append(score)
##################################

##################################原SACのsimulate
# for i in range(n_games):
#         observation = env.reset()
#         done = False
#         score = 0
#         while not done:
#             action = MECSnet.choose_action(observation)
#             observation_, reward, done, info = env.step(action)
#             score += reward
#             MECSnet.remember(observation, action, reward, observation_, done)
#
#             MECSnet.learn()
#             observation = observation_
#         score_history.append(score)
#         avg_score = np.mean(score_history[-100:])
#
#
#
#         print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
#         list.append(score)
##########################################





print(list)
plt.plot(np.arange(len(list)),list)
plt.xlabel('time')
plt.ylabel('formatReward')

plt.grid()
plt.savefig('figure2.eps', format='eps', dpi=1000)
plt.show()


