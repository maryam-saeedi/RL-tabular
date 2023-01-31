#%% import
import numpy as np
from baseModel import Base

class QLearning(Base):
    def __init__(self, env, seed, df, lr, epslilon, func=None):
        super().__init__(env, seed, df, lr, epslilon, func)

    def generateEpisode(self):
        super().generateEpisode()
        iterations = 0
        totral_reward = 0
        current_state, _ = self.env.reset(seed = self.seed)
        while True:
            iterations+=1
            bestAction = self.getAction(current_state)
            next_state,reward,done,_,_ = self.env.step(bestAction)
            totral_reward+=reward
            q = self.LEARNING_RATE * (reward + self.DISCOUNT * np.max(self.Q[next_state,:]) - self.Q[current_state,bestAction])
            self.Q[current_state,bestAction]+=q
            current_state = next_state

            if done:
                break

        return totral_reward/iterations

