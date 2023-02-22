import numpy as np
import matplotlib.pyplot as plt
import imageio

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
            reward = self.generateEpisode()
            rewards.append(reward)
        return rewards
            
    def reset(self):
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.epsilon = self.EPSILON

    def run(self, repeats, episodes):
        agents_values = np.empty((0, self.env.observation_space.n, self.env.action_space.n))
        rewards = []
        for rep in range(repeats):
            print(rep)
            self.reset()
            r = self.repeat(episodes)
            agents_values = np.append(agents_values, [self.Q], 0)
            r = self.slidingWindowAR(r)
            rewards.append(r)
        rewards = np.array(rewards)
        rewards = np.array(rewards)
        AR = np.mean(rewards, axis=0)
        self.AQ = np.mean(agents_values, 0)
        return AR

    def slidingWindowAR(self, reward, WINDOW=25, SLIDE=5):
        AR = []
        for start in range(0, len(reward)-WINDOW, SLIDE):
            AR.append(np.mean(reward[start:start+WINDOW]))
        return AR

    def test(self, max_steps=10000, render=False, saveGIF=None):
        total_reward = 0
        iteration = 0
        state, _ = self.env.reset(seed = self.seed)
        frames = []
        for _ in range(max_steps):
            if saveGIF:
                f = self.env.render()
                frames.append(f)

            if render:      # Render and display current state of the environment
                plt.imshow(self.env.render()) # render current state and pass to pyplot
                plt.axis('off')
                plt.draw()
                plt.pause(0.1)

            
            
            action = np.argmax(self.AQ[state])      # take action with greedy policy w.r.t average Q-values
            state,reward,done,_,_ = self.env.step(action)     # take action and observe reward and next state
            total_reward+=reward
            iteration+=1
            if done:
                if saveGIF:
                    f = self.env.render()
                    frames.append(f)
                if render:      # Render and display current state of the environment
                    plt.imshow(self.env.render()) # render current state and pass to pyplot
                    plt.axis('off')
                    plt.draw()
                    plt.pause(0.1)
                break
        plt.close()

        if saveGIF:
            imageio.mimsave(saveGIF, frames, fps = 5, loop=1)

        return total_reward, iteration
