import numpy as np
from tqdm import tqdm  # 用于显示美观的进度条
from core.tcs_environment import TrainControlEnvironment
from core.performance_monitor import PerformanceMonitor
from utils.agent_factory import create_agent


class FspSimulator:
    """
    Manages the simulation loop for a single-agent reinforcement learning setup.
    It coordinates the interaction between the agent and the environment,
    and collects data using a performance monitor.
    """

    def __init__(self, config, logger):
        """
        Initializes the simulator.
        :param config: The global configuration object.
        :param logger: The logger instance for recording messages.
        """
        self.config = config
        self.logger = logger
        self.logger.info("Initializing the simulation environment and agent...")

        # 1. 创建列车控制系统环境
        self.env = TrainControlEnvironment(config, logger)

        # 2. 使用智能体工厂创建Q-learning智能体
        self.defender_agent = create_agent(config, agent_type='q_learning')

        # 3. 初始化性能监控器
        self.monitor = PerformanceMonitor()

        self.logger.info("Simulator initialized successfully.")

    def run_simulation(self):
        """
        Executes the main training loop for the specified number of episodes.
        """
        sim_params = self.config.simulation
        self.logger.info(f"Starting simulation for {sim_params.num_episodes} episodes.")

        # 使用tqdm创建一个进度条，让训练过程可视化
        for episode in tqdm(range(sim_params.num_episodes), desc="Training Progress"):
            # 每个回合开始时，重置环境
            current_state = self.env.reset()

            for step in range(sim_params.max_steps_per_episode):
                # 1. 智能体根据当前状态选择一个行动
                defender_action = self.defender_agent.choose_action(current_state)

                # 2. 环境执行该行动，并返回结果
                next_state, reward_info, done = self.env.step(defender_action)

                # 3. 智能体从这次经验中学习
                total_reward = reward_info['total_reward']
                self.defender_agent.learn(current_state, defender_action, total_reward, next_state)

                # 4. 记录这一步的所有数据
                self.monitor.record(
                    episode=episode,
                    step=step,
                    defender_action=np.argmax(self.env.defender_action_vector),
                    attacker_action=reward_info['attacker_action'],
                    outcome=reward_info['outcome'],
                    total_reward=total_reward,
                    extrinsic_reward=reward_info['extrinsic_reward'],
                    intrinsic_reward=reward_info['intrinsic_reward'],
                    shaping_reward=reward_info['shaping_reward'],
                    system_risk=reward_info['current_risk'],
                    epsilon=self.defender_agent.epsilon
                )

                # 5. 更新状态
                current_state = next_state

                if done:
                    break  # 在这个设定下，done通常是False，但保留以备将来扩展

            # 每个回合结束后，衰减epsilon值，以减少探索
            self.defender_agent.decay_epsilon()

        self.logger.info("Simulation finished.")
        return self.monitor.get_dataframe()

