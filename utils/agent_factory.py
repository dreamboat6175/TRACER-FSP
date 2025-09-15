from agents.q_learning_agent import QLearningAgent


# If you add more agents in the future, you can import them here.
# from agents.sarsa_agent import SarsaAgent
# from agents.double_q_learning_agent import DoubleQLearningAgent

def create_agent(config, agent_type='q_learning'):
    """
    Factory function to create and return a learning agent based on the specified type.

    :param config: The global configuration object (SimpleNamespace).
    :param agent_type: A string specifying the type of agent to create ('q_learning', 'sarsa', etc.).
    :return: An instance of the specified agent.
    """

    agent_params = config.agent_params
    num_actions = config.game_setup.num_defender_actions

    if agent_type == 'q_learning':
        return QLearningAgent(
            agent_id='defender',
            num_actions=num_actions,
            learning_rate=agent_params.learning_rate,
            discount_factor=config.simulation.discount_factor,
            exploration_rate=agent_params.exploration_rate,
            min_exploration_rate=agent_params.min_exploration_rate,
            exploration_decay_rate=agent_params.exploration_decay_rate
        )
    # You can add more agent types here as the project grows.
    # elif agent_type == 'sarsa':
    #     return SarsaAgent(...)
    # elif agent_type == 'double_q_learning':
    #     return DoubleQLearningAgent(...)
    else:
        raise ValueError(f"Unknown agent type specified: {agent_type}")

