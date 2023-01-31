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


#%% Q-Learning
qLearning_se = QLearning(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, EPSILON)     #static epsilon
meanAR_qs = qLearning_se.run(REPS, EPISODES)

qLearning_de = QLearning(env, STUDENT_NUM, DISCOUNT, LEARNING_RATE, 0.9, lambda x: x - 0.002*x)     #decreasing epsilon
meanAR_qd = qLearning_de.run(REPS, EPISODES)

plt.plot(meanAR_qs)
plt.plot(meanAR_qd)
plt.title("Q-Learning Average Reward")
plt.legend(["epsilon=0.1", "decreasing epsilon"])      
plt.xlabel("episod")
plt.ylabel("AR")
plt.show()


