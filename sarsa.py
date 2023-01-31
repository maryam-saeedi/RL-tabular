#%% import
import numpy as np
from baseModel import Base

class SARSA(Base):
    def __init__(self, env, seed, df, lr, epslilon, n, func=None):
        super().__init__(env, seed, df, lr, epslilon, func)
        self.STEP = n

    def generateEpisode(self):
        super().generateEpisode()
        iterations = 0
        totral_reward = 0
        S = []
        A = []
        R = []

        state, _ = self.env.reset(seed = self.seed)
        S.append(state)
        action = self.getAction(state)
        A.append(action)

        T = np.inf
        t = 0

        while True:     # loop for each state of episode
            if t<T:
                iterations+=1
                state,reward,done,_,_ = self.env.step(action)     # take action and observe reward and next state
                totral_reward+=reward
                S.append(state)
                R.append(reward)

                if done:
                    T = t+1
                else:
                    ## Select and store an action At+1
                    action = self.getAction(state)
                    A.append(action)
            tau = t-self.STEP+1
            if tau >= 0:
                G = np.sum([self.DISCOUNT**(i-tau-1) * R[i-1] for i in range(tau+1,min(tau+self.STEP,T)+1)])
                if tau+self.STEP < T:
                    G+= (self.DISCOUNT**self.STEP * self.Q[S[tau+self.STEP], A[tau+self.STEP]])
                self.Q[S[tau], A[tau]] += (self.LEARNING_RATE * (G-self.Q[S[tau],A[tau]]))

            if tau == T-1:
                break
                
            t+=1
        return totral_reward/iterations