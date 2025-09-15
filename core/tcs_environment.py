# core/enhanced_tcs_environment.py
import numpy as np
import random
from collections import defaultdict


class IntelligentAttacker:
    """
    Intelligent attacker that learns and adapts strategies based on defender behavior.
    """

    def __init__(self, config):
        self.config = config
        self.num_actions = config.game_setup.num_attacker_actions

        # Strategy learning components
        self.success_history = defaultdict(list)
        self.defender_pattern_memory = defaultdict(int)
        self.attack_success_rate = defaultdict(float)

        # Adaptive parameters
        self.exploration_rate = 0.3
        self.learning_rate = 0.1
        self.memory_window = 100

    def observe_defender_action(self, defender_action, outcome):
        """Observe and learn from defender actions and outcomes."""
        self.defender_pattern_memory[defender_action] += 1

        # Update success rates for different attack-defense combinations
        if outcome in ['FN']:  # Successful attack
            self.attack_success_rate[defender_action] = (
                    self.attack_success_rate[defender_action] * 0.9 + 0.1 * 1.0)
        elif outcome in ['TP']:  # Failed attack
            self.attack_success_rate[defender_action] = (
                    self.attack_success_rate[defender_action] * 0.9 + 0.1 * 0.0)

    def choose_attack_strategy(self, episode, recent_defender_actions):
        """
        Choose attack strategy based on learned defender patterns.
        """
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Explore: random attack or no attack
            if random.random() < 0.7:  # 70% chance to attack when exploring
                return random.randint(0, self.num_actions - 1)
            else:
                return None

        # Exploitation: attack the most vulnerable defended asset
        if not recent_defender_actions:
            return random.randint(0, self.num_actions - 1) if random.random() < 0.5 else None

        # Analyze recent defender patterns
        defender_freq = defaultdict(int)
        for action in recent_defender_actions[-self.memory_window:]:
            defender_freq[action] += 1

        # Find most frequently defended asset
        most_defended = max(defender_freq, key=defender_freq.get)

        # Choose attack that exploits this defense pattern
        # Prefer attacks on less defended assets
        asset_defense_levels = defaultdict(int)
        for def_action in recent_defender_actions[-20:]:  # Recent 20 actions
            asset_idx = def_action // self.config.game_setup.defender_actions_per_asset
            asset_defense_levels[asset_idx] += 1

        # Attack the least defended asset
        if asset_defense_levels:
            least_defended_asset = min(asset_defense_levels, key=asset_defense_levels.get)
            # Choose a random attack type for that asset
            attack_actions_for_asset = [
                i for i in range(self.num_actions)
                if i // self.config.game_setup.attacker_actions_per_asset == least_defended_asset
            ]
            return random.choice(attack_actions_for_asset)
        else:
            return random.randint(0, self.num_actions - 1)


class EnhancedTrainControlEnvironment:
    """
    Enhanced environment with intelligent attacker and comprehensive state modeling.
    """

    def __init__(self, config):
        self.config = config
        self.gs = config.game_setup

        # Initialize intelligent attacker
        self.attacker = IntelligentAttacker(config)

        # Enhanced state representation
        self.state = np.zeros(self.gs.num_defender_actions)
        self.system_health = 1.0
        self.recent_attacks = []
        self.performance_metrics = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'false_alarms': 0,
            'detection_rate': 0.0
        }

        # Track action history for intelligent attacker
        self.defender_action_history = []

    def reset(self):
        """Reset environment to initial state."""
        self.state = np.zeros(self.gs.num_defender_actions)
        self.state[0] = 1  # Default to first defense strategy
        return self.state.copy()

    def step(self, defender_action_idx):
        """
        Enhanced step function with intelligent attacker and comprehensive metrics.
        """
        # Store defender action for attacker learning
        self.defender_action_history.append(defender_action_idx)

        # Intelligent attacker decision
        attacker_action_idx = self.attacker.choose_attack_strategy(
            len(self.defender_action_history), self.defender_action_history)

        # Determine outcome
        is_attack = attacker_action_idx is not None
        outcome = self._determine_outcome(attacker_action_idx, defender_action_idx)

        # Let attacker learn from the outcome
        if is_attack:
            self.attacker.observe_defender_action(defender_action_idx, outcome)

        # Update system state and metrics
        self._update_system_metrics(outcome, is_attack)

        # Update defender state
        next_state = np.zeros_like(self.state)
        next_state[defender_action_idx] = 1
        self.state = next_state

        # Calculate comprehensive reward
        reward_info = self._calculate_comprehensive_reward(
            outcome, attacker_action_idx, defender_action_idx)

        return self.state.copy(), reward_info, False

    def _determine_outcome(self, attacker_action_idx, defender_action_idx):
        """Enhanced outcome determination with dynamic probabilities."""
        if attacker_action_idx is None:
            # No attack scenario
            base_fa_prob = self.config.model_params.false_alarm_probability[defender_action_idx]

            # Adjust false alarm probability based on system stress
            stress_factor = 1.0 - self.system_health
            adjusted_fa_prob = base_fa_prob * (1 + stress_factor)

            if random.random() < adjusted_fa_prob:
                return "FP"  # False Positive
            else:
                return "TN"  # True Negative
        else:
            # Attack scenario
            base_detection_prob = self.config.model_params.detection_probability[
                attacker_action_idx][defender_action_idx]

            # Adjust detection probability based on recent performance
            performance_factor = self.performance_metrics['detection_rate']
            adjusted_detection_prob = base_detection_prob * (0.8 + 0.4 * performance_factor)
            adjusted_detection_prob = max(0.1, min(0.95, adjusted_detection_prob))

            if random.random() < adjusted_detection_prob:
                return "TP"  # True Positive
            else:
                return "FN"  # False Negative

    def _update_system_metrics(self, outcome, is_attack):
        """Update comprehensive system performance metrics."""
        if is_attack:
            self.performance_metrics['total_attacks'] += 1
            if outcome == "FN":
                self.performance_metrics['successful_attacks'] += 1
                self.system_health = max(0.0, self.system_health - 0.05)
            elif outcome == "TP":
                self.system_health = min(1.0, self.system_health + 0.01)

        if outcome == "FP":
            self.performance_metrics['false_alarms'] += 1
            self.system_health = max(0.0, self.system_health - 0.01)

        # Update detection rate
        total_attacks = self.performance_metrics['total_attacks']
        if total_attacks > 0:
            successful_detections = total_attacks - self.performance_metrics['successful_attacks']
            self.performance_metrics['detection_rate'] = successful_detections / total_attacks

        # Track recent attacks for pattern analysis
        if is_attack:
            self.recent_attacks.append(outcome)
            if len(self.recent_attacks) > 50:
                self.recent_attacks.pop(0)

    def _calculate_comprehensive_reward(self, outcome, attacker_action_idx, defender_action_idx):
        """Calculate comprehensive reward incorporating multiple factors."""
        from core.tcs_environment import RewardCalculator

        reward_calc = RewardCalculator(self.config)

        # Basic payoffs
        attacker_payoff, defender_payoff = reward_calc.get_payoffs(
            outcome, attacker_action_idx, defender_action_idx)

        # Performance-based modifiers
        performance_bonus = self._calculate_performance_bonus(outcome)
        system_health_penalty = (1.0 - self.system_health) * -10

        # Adaptivity bonus - reward for handling diverse attacks
        adaptivity_bonus = self._calculate_adaptivity_bonus()

        total_reward = defender_payoff + performance_bonus + system_health_penalty + adaptivity_bonus

        return {
            'total_reward': total_reward,
            'base_payoff': defender_payoff,
            'performance_bonus': performance_bonus,
            'health_penalty': system_health_penalty,
            'adaptivity_bonus': adaptivity_bonus,
            'outcome': outcome,
            'attacker_action': attacker_action_idx,
            'system_health': self.system_health,
            'detection_rate': self.performance_metrics['detection_rate']
        }

    def _calculate_performance_bonus(self, outcome):
        """Calculate performance-based reward bonus."""
        if outcome == "TP":
            return 5  # Successful detection
        elif outcome == "TN":
            return 1  # Normal operation
        elif outcome == "FN":
            return -10  # Missed attack (severe penalty)
        elif outcome == "FP":
            return -3  # False alarm (moderate penalty)
        return 0

    def _calculate_adaptivity_bonus(self):
        """Reward adaptive behavior against diverse attack patterns."""
        if len(self.defender_action_history) < 20:
            return 0

        recent_actions = self.defender_action_history[-20:]
        unique_actions = len(set(recent_actions))
        max_unique = min(20, self.gs.num_defender_actions)

        # Bonus for using diverse defense strategies
        diversity_ratio = unique_actions / max_unique
        return diversity_ratio * 2