# core/enhanced_fsp_simulator.py
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import random
from core.tcs_environment import TrainControlEnvironment
from core.performance_monitor import PerformanceMonitor
from utils.agent_factory import create_agent


class FictitiousSelfPlaySimulator:
    """
    Enhanced simulator implementing true Fictitious Self-Play for 
    train control system security optimization.
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        # Initialize environment and reward calculator
        self.env = TrainControlEnvironment(config)

        # Create defender agent
        self.defender_agent = create_agent(config, agent_type='q_learning')

        # FSP-specific components
        self.attacker_strategy_history = defaultdict(list)
        self.defender_strategy_history = defaultdict(list)
        self.attack_probability = config.simulation.attack_probability

        # Performance monitoring
        self.monitor = PerformanceMonitor()

        # Enhanced state representation
        self.state_dimension = self._calculate_state_dimension()

        self.logger.info("Enhanced FSP Simulator initialized successfully.")

    def _calculate_state_dimension(self):
        """Calculate the enhanced state space dimension."""
        base_dim = self.config.game_setup.num_defender_actions
        # Add system status, threat level, resource utilization
        enhanced_dim = base_dim + 3  # +3 for additional state features
        return enhanced_dim

    def _get_enhanced_state(self, base_state, system_status=0, threat_level=0, resource_util=0):
        """Create enhanced state representation including system context."""
        enhanced_state = np.zeros(self.state_dimension)
        enhanced_state[:len(base_state)] = base_state
        enhanced_state[-3] = system_status
        enhanced_state[-2] = threat_level
        enhanced_state[-1] = resource_util
        return enhanced_state

    def _sample_attacker_strategy(self, episode):
        """
        Sample attacker action based on historical strategy distribution.
        This implements the core FSP mechanism.
        """
        if episode == 0 or not self.attacker_strategy_history:
            # Initial random strategy
            if random.random() < self.attack_probability:
                return random.randint(0, self.config.game_setup.num_attacker_actions - 1)
            else:
                return None

        # Compute empirical strategy based on history
        total_actions = sum(len(history) for history in self.attacker_strategy_history.values())

        if total_actions == 0:
            return None if random.random() > self.attack_probability else random.randint(0,
                                                                                         self.config.game_setup.num_attacker_actions - 1)

        # Sample based on empirical frequency
        if random.random() < self.attack_probability:
            all_actions = []
            for actions in self.attacker_strategy_history.values():
                all_actions.extend([a for a in actions if a is not None])

            if all_actions:
                return random.choice(all_actions)
            else:
                return random.randint(0, self.config.game_setup.num_attacker_actions - 1)
        else:
            return None

    def _update_strategy_history(self, episode, attacker_action, defender_action):
        """Update strategy histories for FSP."""
        self.attacker_strategy_history[episode].append(attacker_action)
        self.defender_strategy_history[episode].append(defender_action)

    def _calculate_strategy_diversity_bonus(self, episode):
        """Calculate bonus for maintaining strategic diversity."""
        if episode < 10:
            return 0

        recent_episodes = max(1, episode - 50)
        recent_actions = []
        for ep in range(recent_episodes, episode):
            if ep in self.defender_strategy_history:
                recent_actions.extend(self.defender_strategy_history[ep])

        if not recent_actions:
            return 0

        # Calculate entropy of recent actions
        action_counts = defaultdict(int)
        for action in recent_actions:
            action_counts[action] += 1

        total_actions = len(recent_actions)
        entropy = 0
        for count in action_counts.values():
            prob = count / total_actions
            entropy -= prob * np.log2(prob + 1e-8)

        # Normalize entropy and provide bonus
        max_entropy = np.log2(self.config.game_setup.num_defender_actions)
        return 0.1 * (entropy / max_entropy)

    def run_simulation(self):
        """
        运行仿真主循环。
        此版本已适配内部包含攻击者和奖励计算的增强版环境。
        """
        # 从配置文件加载仿真参数
        sim_params = self.config.simulation
        self.logger.info(f"开始仿真，共 {sim_params.num_episodes} 个回合。")

        # 初始化用于存储高级指标的列表（当前版本暂时为空）
        convergence_metrics = []
        strategy_evolution = []

        # 使用tqdm库创建进度条，迭代所有回合
        for episode in tqdm(range(sim_params.num_episodes), desc="FSP 训练进度"):
            # 1. 重置环境，获取初始状态
            current_state = self.env.reset()

            # 初始化回合结束标志
            done = False

            # 在当前回合内执行多个步骤
            for step in range(sim_params.max_steps_per_episode):
                # 2. 防御者Agent根据当前状态选择一个动作
                defender_action = self.defender_agent.choose_action(current_state)

                # 3. 环境执行步骤，仅传入防御者的动作
                #    环境内部会自动处理攻击者的动作、计算奖励并返回所有信息
                next_state, reward_info, done = self.env.step(defender_action)

                # 4. 从环境返回的 `reward_info` 字典中提取关键信息
                total_reward = reward_info['total_reward']
                outcome = reward_info['outcome']
                # 使用 .get() 安全地获取攻击者动作，因为它可能不存在（即没有发生攻击）
                attacker_action = reward_info.get('attacker_action')

                # 5. 让防御者Agent根据经验进行学习
                self.defender_agent.learn(
                    current_state,
                    defender_action,
                    total_reward,
                    next_state
                )

                # 6. 使用性能监视器记录此步骤的详细数据
                self.monitor.record(
                    episode=episode,
                    step=step,
                    defender_action=defender_action,
                    attacker_action=attacker_action,
                    outcome=outcome,
                    total_reward=total_reward,
                    defender_payoff=reward_info.get('base_payoff'),
                    system_risk=1 - reward_info.get('system_health', 1.0),  # 将健康度转化为风险值
                    epsilon=self.defender_agent.epsilon,
                )

                # 7. 更新用于分析的策略历史记录
                self._update_strategy_history(episode, attacker_action, defender_action)

                # 8. 将状态更新为下一步的状态，准备下一次迭代
                current_state = next_state

                # 如果环境发出了结束信号，则提前终止当前回合
                if done:
                    break

            # 9. 每个回合结束后，调用Agent的函数来降低探索率（epsilon）
            self.defender_agent.decay_epsilon()

            # 10. 定期（每500个回合）打印日志，监控训练进展
            if episode % 500 == 0 and episode > 0:
                # 从已记录的数据中查询当前回合的奖励数据
                recent_rewards = self.monitor.get_dataframe().query(f'episode == {episode}')['total_reward']
                if not recent_rewards.empty:
                    avg_reward = recent_rewards.mean()
                    self.logger.info(f"回合 {episode}: 平均奖励 = {avg_reward:.4f}, "
                                     f"Epsilon = {self.defender_agent.epsilon:.4f}")

        self.logger.info("FSP 仿真完成。")

        # 仿真结束后，获取所有记录的数据
        results_df = self.monitor.get_dataframe()

        # 返回结果
        return results_df, convergence_metrics, strategy_evolution

    def _assess_threat_level(self, episode):
        """Dynamic threat level assessment based on recent attack patterns."""
        if episode < 50:
            return random.uniform(0.3, 0.7)

        recent_attacks = 0
        recent_episodes = 50
        start_episode = max(0, episode - recent_episodes)

        for ep in range(start_episode, episode):
            if ep in self.attacker_strategy_history:
                attacks = [a for a in self.attacker_strategy_history[ep] if a is not None]
                recent_attacks += len(attacks)

        # Normalize threat level
        max_possible_attacks = recent_episodes * self.config.simulation.max_steps_per_episode
        threat_level = min(1.0, recent_attacks / (max_possible_attacks * self.attack_probability))

        return threat_level

    def _analyze_episode_convergence(self, episode, rewards, actions, convergence_metrics):
        """Analyze convergence properties of the current episode."""
        if episode < 100:
            return

        # Calculate reward variance over recent episodes
        window_size = 50
        if episode >= window_size:
            recent_rewards = rewards[-window_size:]
            reward_variance = np.var(recent_rewards)

            # Strategy stability metric
            defender_actions = [a[1] for a in actions]
            strategy_entropy = self._calculate_action_entropy(defender_actions)

            convergence_metrics.append({
                'episode': episode,
                'reward_variance': reward_variance,
                'strategy_entropy': strategy_entropy,
                'mean_reward': np.mean(recent_rewards)
            })

    def _track_strategy_evolution(self, episode, actions, strategy_evolution):
        """Track evolution of strategies over time."""
        if episode % 100 == 0 and episode > 0:
            defender_actions = [a[1] for a in actions]
            attacker_actions = [a[0] for a in actions if a[0] is not None]

            defender_dist = self._calculate_action_distribution(
                defender_actions, self.config.game_setup.num_defender_actions)
            attacker_dist = self._calculate_action_distribution(
                attacker_actions, self.config.game_setup.num_attacker_actions)

            strategy_evolution.append({
                'episode': episode,
                'defender_distribution': defender_dist,
                'attacker_distribution': attacker_dist
            })

    def _calculate_action_entropy(self, actions):
        """Calculate entropy of action distribution."""
        if not actions:
            return 0

        action_counts = defaultdict(int)
        for action in actions:
            action_counts[action] += 1

        total_actions = len(actions)
        entropy = 0
        for count in action_counts.values():
            prob = count / total_actions
            entropy -= prob * np.log2(prob + 1e-8)

        return entropy

    def _calculate_action_distribution(self, actions, num_actions):
        """Calculate normalized action distribution."""
        if not actions:
            return [0] * num_actions

        distribution = [0] * num_actions
        for action in actions:
            if action < num_actions:
                distribution[action] += 1

        total = sum(distribution)
        if total > 0:
            distribution = [count / total for count in distribution]

        return distribution