import numpy as np
from baseModel import Base

class OffPolicyMC(Base):
    def __init__(self, env, policy, seed, discount, lr, epsilon, func=None):
        super().__init__(env, seed, discount, lr, epsilon, func)
        self.behavioralPolicy = policy
        self.C = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def generateEpisode(self):
        super().generateEpisode()
        iterations = 0
        totral_reward = 0
        S = []
        A = []
        R = []
        state, _ = self.env.reset(seed = self.seed)
        S.append(state)

        # generate episode following behavioral policy
        while True:
            iterations+=1
            action = self.behavioralPolicy.takeAction(state)
            A.append(action)
            state,reward,done,_,_ = self.env.step(action)     # take action and observe reward and next state
            R.append(reward)
            totral_reward+=reward
            
            if done:
                break
            S.append(state)

        G = 0
        W = 1
        for t in range(len(S)-1, -1, -1):
            G = self.DISCOUNT * G + R[t]
            self.C[S[t], A[t]] += W
            self.Q[S[t], A[t]] += (W/self.C[S[t], A[t]]) * (G - self.Q[S[t], A[t]])
            if self.getAction(S[t]) != A[t]:
                break
            self.behavioralPolicy.update(self.Q)
            W = W*(self.getPolicy(S[t], A[t])/self.behavioralPolicy.stateActionProbaility(S[t], A[t]))


        return totral_reward/iterations   


