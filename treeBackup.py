import numpy as np
from baseModel import Base

class TreeBackup(Base):
    def __init__(self, env, seed, discount, lr, epsilon, n, func=None):
        super().__init__(env, seed, discount, lr, epsilon, func)
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
                if t+1 >= T:
                    G = R[T-1]
                else:
                    G = R[t] + self.DISCOUNT*np.sum([self.getPolicy(S[t+1], a)*self.Q[S[t+1],a] for a in range(self.env.action_space.n)])
                for k in range(min(t,T-1), tau, -1):
                    G_new = R[k-1]
                    for a in range(self.env.action_space.n):
                        if a!=A[k]:
                            G_new+=self.DISCOUNT*self.getPolicy(S[k],a)*self.Q[S[k],a]
                    G_new+=self.DISCOUNT*self.getPolicy(S[k],A[k])*G
                    G = G_new
                Q[S[tau], A[tau]] += (self.LEARNING_RATE * (G-self.Q[S[tau],A[tau]]))

            if tau == T-1:
                break
                
            t+=1
            
        return totral_reward/iterations