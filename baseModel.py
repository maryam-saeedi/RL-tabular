import numpy as np

class Base:
    def __init__(self, env, seed, discount, lr, epsilon, func=None):
        self.env = env
        self.seed = seed
        self.EPSILON = epsilon
        self.func = func
        self.DISCOUNT = discount
        self.LEARNING_RATE = lr
        
    def updateEpsilon(self):
        if self.func:
            self.epsilon = self.func(self.epsilon)
        
    def getAction(self, state):
        if np.random.uniform(0,1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[state,:])
        return action

    def getPolicy(self, state, action):
        '''
        return probability of doing action in state based on e-greedy policy
        '''
        if action == np.argmax(self.Q[state,:]):
            return 1-self.epsilon+(self.epsilon/self.env.action_space.n)
        return self.epsilon/self.env.action_space.n

    def generateEpisode(self):
        self.updateEpsilon()

    def repeat(self, episodes):
        rewards = []
        for episode in range(episodes):
            print("episode", episode)
            reward = self.generateEpisode()
            rewards.append(reward)
        return rewards
            
    def reset(self):
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.epsilon = self.EPSILON

    def run(self, repeats, episodes):
        rewards = []
        for rep in range(repeats):
            print("rep", rep)
            self.reset()
            r = self.repeat(episodes)
            r = self.slidingWindowAR(r)
            rewards.append(r)
        rewards = np.array(rewards)
        rewards = np.array(rewards)
        return np.mean(rewards, axis=0)

    def slidingWindowAR(self, reward, WINDOW=25, SLIDE=5):
        AR = []
        for start in range(0, len(reward)-WINDOW, SLIDE):
            AR.append(np.mean(reward[start:start+WINDOW]))
        return AR