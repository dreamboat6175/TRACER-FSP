import numpy as np
from agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    An agent that learns to make decisions using the Q-learning algorithm.
    It uses a Q-table to store the expected returns for state-action pairs.
    """

    def __init__(self, agent_id, num_actions, learning_rate, discount_factor, exploration_rate=1.0,
                 min_exploration_rate=0.01, exploration_decay_rate=0.999):
        """
        Initializes the Q-learning agent.
        :param exploration_rate: The initial probability of choosing a random action (epsilon).
        :param min_exploration_rate: The minimum value for epsilon.
        :param exploration_decay_rate: The rate at which epsilon decays after each episode.
        """
        super().__init__(agent_id, num_actions, learning_rate, discount_factor)

        # Q-table: rows represent states, columns represent actions.
        # For this problem, the state is determined by which defense is active.
        # So, the number of states is equal to the number of defense actions.
        self.num_states = num_actions
        self.q_table = np.zeros((self.num_states, self.num_actions))

        # Epsilon-greedy strategy parameters
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay_rate

    def _state_to_index(self, state):
        """
        Converts a one-hot encoded state vector to a single integer index.
        Example: [0, 0, 1, 0] -> 2
        """
        return np.argmax(state)

    def choose_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        - With probability epsilon, choose a random action (explore).
        - With probability 1-epsilon, choose the best-known action (exploit).
        """
        state_idx = self._state_to_index(state)

        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            return np.random.choice(self.num_actions)
        else:
            # Exploitation: choose the action with the highest Q-value for the current state
            return np.argmax(self.q_table[state_idx, :])

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Bellman equation.
        Q(s, a) <- Q(s, a) + alpha * [R + gamma * max(Q(s', a')) - Q(s, a)]
        """
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)

        # Find the maximum Q-value for the next state
        max_next_q_value = np.max(self.q_table[next_state_idx, :])

        # Calculate the TD target
        td_target = reward + self.gamma * max_next_q_value

        # Calculate the TD error
        td_error = td_target - self.q_table[state_idx, action]

        # Update the Q-value for the state-action pair
        self.q_table[state_idx, action] += self.lr * td_error

    def decay_epsilon(self):
        """
        Reduces the exploration rate (epsilon) over time to shift from exploration to exploitation.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

