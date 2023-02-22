#%% import
import numpy as np
import matplotlib.pyplot as plt
import gym
from qLearning import QLearning
from sarsa import SARSA
from treeBackup import TreeBackup
from mc import OffPolicyMC
from policies import *

#%% hyperparameters
REPS = 5
EPISODES = 2000
EPSILON = 0.1
LEARNING_RATE = 0.1
DISCOUNT = 0.9
STUDENT_NUM = 2
STEP = 3
SLIDE = 5
WINDOW = 20

#%%environment
env = gym.make('Taxi-v3', render_mode = 'rgb_array')

# #%% Q-Learning
# qLearning_se = QLearning(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPSILON)     #static epsilon
# meanAR_qs = qLearning_se.run(REPS, EPISODES)
# print("testing...")
# total_r, iters = qLearning_se.test(render=True, saveGIF="qlearning_static_epsilon.gif")
# print(total_r, iters)


# qLearning_de = QLearning(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, 0.9, lambda x: x - 0.002*x)     #decreasing epsilon
# meanAR_qd = qLearning_de.run(REPS, EPISODES)
# print("testing...")
# total_r, iters = qLearning_de.test(render=True, saveGIF="qlearning_descreasing_epsilon.gif")
# print(total_r, iters)

# plt.plot(meanAR_qs)
# plt.plot(meanAR_qd)
# plt.title("Q-Learning Average Reward")
# plt.legend(["epsilon=0.1", "decreasing epsilon"])      
# plt.xlabel("episod")
# plt.ylabel("AR")
# plt.show()


# #%% n-step SARSA
# sarsa_n1 = SARSA(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPSILON, 2)
# meanAR1 = sarsa_n1.run(REPS, EPISODES)
# print("total AR: ", np.mean(meanAR1))
# print("testing...")
# total_r, iters = sarsa_n1.test(render=True, saveGIF="sarsa_n2.gif")
# print(total_r, iters)

# sarsa_n2 = SARSA(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPSILON, 5)
# meanAR2 = sarsa_n2.run(REPS, EPISODES)
# print("total AR: ", np.mean(meanAR2))
# print("testing...")
# total_r, iters = sarsa_n2.test(render=True, saveGIF="sarsa_n5.gif")
# print(total_r, iters)

# sarsa_n3 = SARSA(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPSILON, 10)
# meanAR3 = sarsa_n3.run(REPS, EPISODES)
# print("total AR: ", np.mean(meanAR3))
# print("testing...")
# total_r, iters = sarsa_n3.test(render=True, saveGIF="sarsa_n10.gif")
# print(total_r, iters)

# plt.plot(meanAR1)
# plt.plot(meanAR2)
# plt.plot(meanAR3)
# plt.title("n-step SARSA Average Reward")
# plt.legend(["n=2", "n=5", "n=10"])      
# plt.xlabel("episod")
# plt.ylabel("AR")
# plt.show()

# #%% n-step tree backup
# tb_n1 = TreeBackup(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPSILON, 2)
# meanARt1 = tb_n1.run(REPS, EPISODES)
# print("total AR: ", np.mean(meanARt1))
# print("testing...")
# total_r, iters = tb_n1.test(render=True, saveGIF="tree_backup_n2.gif")
# print(total_r, iters)

# tb_n2 = TreeBackup(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPSILON, 5)
# meanARt2 = tb_n2.run(REPS, EPISODES)
# print("total AR: ", np.mean(meanARt2))
# print("testing...")
# total_r, iters = tb_n2.test(render=True, saveGIF="tree_backup_n5.gif")
# print(total_r, iters)

# tb_n3 = TreeBackup(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPSILON, 10)
# meanARt3 = tb_n3.run(REPS, EPISODES)
# print("total AR: ", np.mean(meanARt3))
# print("testing...")
# total_r, iters = tb_n3.test(render=True, saveGIF="tree_backup_n10.gif")
# print(total_r, iters)

# plt.plot(meanARt1)
# plt.plot(meanARt2)
# plt.plot(meanARt3)
# plt.title("n-step Tree Backup Average Reward")
# plt.legend(["n=2", "n=5", "n=10"])      
# plt.xlabel("episod")
# plt.ylabel("AR")
# plt.show()

#%% off-policy monte carlo
mc_se = OffPolicyMC(env, EGreedyPolicy(0.5, env.observation_space.n, env.action_space.n), STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPSILON)
meanAR_ms = mc_se.run(REPS, EPISODES)
print("total AR: ", np.mean(meanAR_ms))
print("testing...")
total_r, iters = mc_se.test(render=True, saveGIF="mc_static_epsilon.gif")
print(total_r, iters)

mc_de = OffPolicyMC(env, EGreedyPolicy(0.5, env.observation_space.n, env.action_space.n), STUDENT_NUM, DISCOUNT, LEARNING_RATE, 0.9, lambda x: x - 0.005*x)
meanAR_md = mc_de.run(REPS, EPISODES)
print("total AR: ", np.mean(meanAR_md))
print("testing...")
total_r, iters = mc_de.test(render=True, saveGIF="mc_decreasing_epsilon.gif")
print(total_r, iters)

plt.plot(meanAR_ms)
plt.plot(meanAR_md)
plt.title("Off-policy MC Average Reward")
plt.legend(["epsilon=0.1", "decreasing epsilon"])      
plt.xlabel("episod")
plt.ylabel("AR")
plt.show()
