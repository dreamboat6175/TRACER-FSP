class BaseAgent:
    """
    An abstract base class for all learning agents.
    It defines the common interface that all agents must implement.
    """

    def __init__(self, agent_id, num_actions, learning_rate, discount_factor):
        """
        Initializes the base agent.
        :param agent_id: A unique identifier for the agent.
        :param num_actions: The size of the action space.
        :param learning_rate: The learning rate (alpha) for the agent.
        :param discount_factor: The discount factor (gamma) for future rewards.
        """
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor

    def choose_action(self, state):
        """
        Selects an action based on the current state.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by a subclass.")

    def learn(self, state, action, reward, next_state):
        """
        Updates the agent's knowledge based on an experience.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by a subclass.")

