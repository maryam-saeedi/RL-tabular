#%% import
import numpy as np
import matplotlib.pyplot as plt
import gym
from qLearning import QLearning
from sarsa import SARSA

#%% hyperparameters
REPS = 20
EPISODES = 2000
EPSILON = 0.1
LEARNING_RATE = 0.1
DISCOUNT = 0.9
STUDENT_NUM = 63
STEP = 3
SLIDE = 5
WINDOW = 20

#%%environment
env = gym.make('Taxi-v3')

#%% get familiar with the environment
print("you can see the environment in each step by render command :")
# env.render()


# #%% Q-Learning
# qLearning_se = QLearning(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPSILON)     #static epsilon
# meanAR_qs = qLearning_se.run(REPS, EPISODES)

# qLearning_de = QLearning(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, 0.9, lambda x: x - 0.002*x)     #decreasing epsilon
# meanAR_qd = qLearning_de.run(REPS, EPISODES)

# plt.plot(meanAR_qs)
# plt.plot(meanAR_qd)
# plt.title("Q-Learning Average Reward")
# plt.legend(["epsilon=0.1", "decreasing epsilon"])      
# plt.xlabel("episod")
# plt.ylabel("AR")
# plt.show()


#%% n-step SARSA
sarsa_n1 = SARSA(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPISODES, 1)
meanAR1 = sarsa_n1.run(REPS, EPISODES)

sarsa_n2 = SARSA(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPISODES, 2)
meanAR2 = sarsa_n2.run(REPS, EPISODES)

sarsa_n3 = SARSA(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPISODES, 3)
meanAR3 = sarsa_n3.run(REPS, EPISODES)

plt.plot(meanAR1)
plt.plot(meanAR2)
plt.plot(meanAR3)
plt.title("n-step SARSA Average Reward")
plt.legend(["n=1", "n=2", "n=3"])      
plt.xlabel("episod")
plt.ylabel("AR")
plt.show()

#%% n-step tree backup
tb_n1 = SARSA(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPISODES, 1)
meanARt1 = tb_n1.run(REPS, EPISODES)

tb_n2 = SARSA(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPISODES, 2)
meanARt2 = tb_n2.run(REPS, EPISODES)

tb_n3 = SARSA(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPISODES, 3)
meanARt3 = tb_n3.run(REPS, EPISODES)

plt.plot(meanARt1)
plt.plot(meanARt2)
plt.plot(meanARt3)
plt.title("n-step Tree Backup Average Reward")
plt.legend(["n=1", "n=2", "n=3"])      
plt.xlabel("episod")
plt.ylabel("AR")
plt.show()