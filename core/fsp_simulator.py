# core/fsp_simulator.py (最终修复版)
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import random
from core.tcs_environment import EnhancedTrainControlEnvironment
from core.performance_monitor import PerformanceMonitor
from utils.agent_factory import create_agent


class FictitiousSelfPlaySimulator:
    """
    模拟器，适配了内部包含攻击者和奖励计算的增强版环境。
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.env = EnhancedTrainControlEnvironment(config)
        self.defender_agent = create_agent(config, agent_type='q_learning')
        self.attacker_strategy_history = defaultdict(list)
        self.defender_strategy_history = defaultdict(list)
        self.monitor = PerformanceMonitor()
        self.logger.info("自适应 FSP 模拟器初始化成功。")

    def run_simulation(self):
        """
        运行仿真主循环。
        """
        sim_params = self.config.simulation
        self.logger.info(f"开始仿真，共 {sim_params.num_episodes} 个回合。")

        convergence_metrics = []
        strategy_evolution = []

        for episode in tqdm(range(sim_params.num_episodes), desc="FSP 训练进度"):
            current_state = self.env.reset()
            done = False

            for step in range(sim_params.max_steps_per_episode):
                defender_action = self.defender_agent.choose_action(current_state)
                next_state, reward_info, done = self.env.step(defender_action)

                total_reward = reward_info['total_reward']
                outcome = reward_info['outcome']
                attacker_action = reward_info.get('attacker_action')

                self.defender_agent.learn(
                    current_state, defender_action,
                    total_reward, next_state)

                # ==============================================================================
                # 解决方案：在这里的 record 调用中，添加 detection_rate
                # ==============================================================================
                self.monitor.record(
                    episode=episode,
                    step=step,
                    defender_action=defender_action,
                    attacker_action=attacker_action,
                    outcome=outcome,
                    total_reward=total_reward,
                    defender_payoff=reward_info.get('base_payoff'),
                    system_risk=1 - reward_info.get('system_health', 1.0),
                    epsilon=self.defender_agent.epsilon,
                    detection_rate=reward_info.get('detection_rate')  # <-- 添加此行
                )

                self._update_strategy_history(episode, attacker_action, defender_action)
                current_state = next_state

                if done:
                    break

            self.defender_agent.decay_epsilon()

            if episode % 500 == 0 and episode > 0:
                recent_rewards = self.monitor.get_dataframe().query(f'episode == {episode}')['total_reward']
                if not recent_rewards.empty:
                    avg_reward = recent_rewards.mean()
                    self.logger.info(f"回合 {episode}: 平均奖励 = {avg_reward:.4f}, "
                                     f"Epsilon = {self.defender_agent.epsilon:.4f}")

        self.logger.info("FSP 仿真完成。")
        results_df = self.monitor.get_dataframe()
        return results_df, convergence_metrics, strategy_evolution

    def _update_strategy_history(self, episode, attacker_action, defender_action):
        """更新策略历史记录。"""
        self.attacker_strategy_history[episode].append(attacker_action)
        self.defender_strategy_history[episode].append(defender_action)