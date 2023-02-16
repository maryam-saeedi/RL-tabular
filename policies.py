import numpy as np

class Policy:
    def __init__(self, state_space_number, action_space_number) -> None:
        self.state_space_number = state_space_number
        self.action_space_number = action_space_number

    def policy(self, state) -> list:
        pass

    def takeAction(self, state) -> int:
        action = np.random.choice(range(self.action_space_number), p=self.policy(state))
        return action

    def stateActionProbaility(self, state, action) -> float:
        return self.policy(state)[action]

    def update(self, Q) -> None:
        pass


class EGreedyPolicy(Policy):
    def __init__(self, epsilon, state_space_number, action_space_number):
        super().__init__(state_space_number, action_space_number)
        self.epsilon = epsilon
        self.Q = np.zeros((state_space_number, action_space_number))

    def policy(self, state) -> list:
        prob = np.ones(self.action_space_number) * self.epsilon / self.action_space_number
        bestAction = np.argmax(self.Q[state])
        prob[bestAction] += (1-self.epsilon)
        return prob

    def update(self, Q) -> None:
        self.Q = Q

class RandomPolicy:
    def __init__(self, state_space_number, action_space_number) -> None:
        self.state_space_number = state_space_number
        self.action_space_number = action_space_number

    def policy(self, state) -> list:
        prob = np.ones(self.action_space_number) / self.action_space_number
        return prob

class GreedyPolicy(Policy):
    def __init__(self, state_space_number, action_space_number) -> None:
        super().__init__(state_space_number, action_space_number)

    def policy(self, state) -> list:
        prob = np.zeros(self.action_space_number)
        bestAction = np.argmax(self.Q[state])
        prob[bestAction] = 1
        return prob

