# core/enhanced_fsp_simulator.py
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import random
from core.tcs_environment import TrainControlEnvironment, RewardCalculator
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
        self.reward_calc = RewardCalculator(config)

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

    def _calculate_adaptive_reward(self, outcome, attacker_action, defender_action,
                                   current_state, next_state, episode):
        """
        Calculate comprehensive reward including FSP-specific components.
        """
        # Basic payoffs
        attacker_payoff, defender_payoff = self.reward_calc.get_payoffs(
            outcome, attacker_action, defender_action)

        # External reward (risk-based)
        ext_reward = self.reward_calc.calculate_external_reward(next_state, attacker_action)

        # Intrinsic reward (exploration)
        int_reward = self.reward_calc.calculate_intrinsic_reward(
            current_state, defender_action, next_state)

        # Shaping reward
        shaping_reward = self.reward_calc.calculate_shaping_reward(current_state, next_state)

        # FSP-specific adaptive component
        strategy_diversity_bonus = self._calculate_strategy_diversity_bonus(episode)

        # Combine rewards
        total_reward = (defender_payoff +
                        self.config.reward_weights.beta_ext * ext_reward +
                        self.config.reward_weights.beta_int * int_reward +
                        shaping_reward +
                        strategy_diversity_bonus)

        return {
            'total_reward': total_reward,
            'defender_payoff': defender_payoff,
            'extrinsic_reward': ext_reward,
            'intrinsic_reward': int_reward,
            'shaping_reward': shaping_reward,
            'strategy_bonus': strategy_diversity_bonus,
            'outcome': outcome,
            'attacker_action': attacker_action,
            'current_risk': self.reward_calc._calculate_cyber_physical_risk(next_state, attacker_action)
        }

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
        Execute the enhanced FSP simulation with comprehensive data collection.
        """
        sim_params = self.config.simulation
        self.logger.info(f"Starting Enhanced FSP simulation for {sim_params.num_episodes} episodes.")

        # Initialize metrics tracking
        convergence_metrics = []
        strategy_evolution = []

        for episode in tqdm(range(sim_params.num_episodes), desc="FSP Training Progress"):
            # Reset environment 
            current_state = self.env.reset() if hasattr(self.env, 'reset') else np.zeros(
                self.config.game_setup.num_defender_actions)

            # Enhanced state representation
            system_status = random.uniform(0, 1)  # Simulated system load
            threat_level = self._assess_threat_level(episode)
            resource_util = random.uniform(0.3, 0.9)  # Resource utilization

            enhanced_state = self._get_enhanced_state(current_state, system_status, threat_level, resource_util)

            episode_rewards = []
            episode_actions = []

            for step in range(sim_params.max_steps_per_episode):
                # FSP attacker strategy sampling
                attacker_action = self._sample_attacker_strategy(episode)

                # Defender action selection
                defender_action = self.defender_agent.choose_action(enhanced_state)
                episode_actions.append((attacker_action, defender_action))

                # Environment step
                outcome, next_base_state = self.env.step(attacker_action, defender_action)

                # Enhanced next state
                next_enhanced_state = self._get_enhanced_state(
                    next_base_state, system_status, threat_level, resource_util)

                # Calculate comprehensive reward
                reward_info = self._calculate_adaptive_reward(
                    outcome, attacker_action, defender_action,
                    enhanced_state, next_enhanced_state, episode)

                episode_rewards.append(reward_info['total_reward'])

                # Agent learning
                self.defender_agent.learn(
                    enhanced_state, defender_action,
                    reward_info['total_reward'], next_enhanced_state)

                # Data recording
                self.monitor.record(
                    episode=episode,
                    step=step,
                    defender_action=defender_action,
                    attacker_action=attacker_action,
                    outcome=outcome,
                    total_reward=reward_info['total_reward'],
                    defender_payoff=reward_info['defender_payoff'],
                    extrinsic_reward=reward_info['extrinsic_reward'],
                    intrinsic_reward=reward_info['intrinsic_reward'],
                    shaping_reward=reward_info['shaping_reward'],
                    strategy_bonus=reward_info['strategy_bonus'],
                    system_risk=reward_info['current_risk'],
                    epsilon=self.defender_agent.epsilon,
                    threat_level=threat_level,
                    system_status=system_status,
                    resource_utilization=resource_util
                )

                # Update strategy histories
                self._update_strategy_history(episode, attacker_action, defender_action)

                # State transition
                enhanced_state = next_enhanced_state

            # Episode-level analysis
            self._analyze_episode_convergence(episode, episode_rewards, episode_actions, convergence_metrics)
            self._track_strategy_evolution(episode, episode_actions, strategy_evolution)

            # Decay exploration
            self.defender_agent.decay_epsilon()

            # Periodic logging
            if episode % 500 == 0:
                self.logger.info(f"Episode {episode}: Avg Reward = {np.mean(episode_rewards):.4f}, "
                                 f"Epsilon = {self.defender_agent.epsilon:.4f}")

        self.logger.info("Enhanced FSP simulation completed.")

        # Add convergence and strategy evolution data to results
        results_df = self.monitor.get_dataframe()
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