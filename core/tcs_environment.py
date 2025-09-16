# core/tcs_environment.py (最终修复版)
import numpy as np
import random
from collections import defaultdict

# ==============================================================================
# 解决方案：在这里重新定义缺失的 RewardCalculator 类
# ==============================================================================
class RewardCalculator:
    """
    一个独立的类，用于根据游戏结果计算支付（payoffs）。
    """
    def __init__(self, config):
        self.config = config
        self.payoffs = config.payoffs

    def get_payoffs(self, outcome, attacker_action, defender_action):
        """
        根据结果（TP, FP, FN, TN）获取攻击者和防御者的支付值。
        """
        # outcome_map = {"TP": 0, "FP": 1, "FN": 2, "TN": 3}
        # outcome_idx = outcome_map.get(outcome, 3) # 默认为 TN

        # 简单实现，直接从配置中查找
        if outcome == "TP":
            attacker_payoff = self.payoffs.TP.attacker
            defender_payoff = self.payoffs.TP.defender
        elif outcome == "FP":
            attacker_payoff = self.payoffs.FP.attacker
            defender_payoff = self.payoffs.FP.defender
        elif outcome == "FN":
            attacker_payoff = self.payoffs.FN.attacker
            defender_payoff = self.payoffs.FN.defender
        else:  # TN
            attacker_payoff = self.payoffs.TN.attacker
            defender_payoff = self.payoffs.TN.defender

        return attacker_payoff, defender_payoff


class IntelligentAttacker:
    """
    能够根据防御者行为学习和调整策略的智能攻击者。
    """
    def __init__(self, config):
        self.config = config
        self.num_actions = config.game_setup.num_attacker_actions
        self.success_history = defaultdict(list)
        self.defender_pattern_memory = defaultdict(int)
        self.attack_success_rate = defaultdict(float)
        self.exploration_rate = 0.3
        self.learning_rate = 0.1
        self.memory_window = 100

    def observe_defender_action(self, defender_action, outcome):
        self.defender_pattern_memory[defender_action] += 1
        if outcome in ['FN']:
            self.attack_success_rate[defender_action] = (
                    self.attack_success_rate[defender_action] * 0.9 + 0.1 * 1.0)
        elif outcome in ['TP']:
            self.attack_success_rate[defender_action] = (
                    self.attack_success_rate[defender_action] * 0.9 + 0.1 * 0.0)

    def choose_attack_strategy(self, episode, recent_defender_actions):
        if random.random() < self.exploration_rate:
            if random.random() < 0.7:
                return random.randint(0, self.num_actions - 1)
            else:
                return None

        if not recent_defender_actions:
            return random.randint(0, self.num_actions - 1) if random.random() < 0.5 else None

        defender_freq = defaultdict(int)
        for action in recent_defender_actions[-self.memory_window:]:
            defender_freq[action] += 1

        most_defended = max(defender_freq, key=defender_freq.get)

        asset_defense_levels = defaultdict(int)
        for def_action in recent_defender_actions[-20:]:
            asset_idx = def_action // self.config.game_setup.defender_actions_per_asset
            asset_defense_levels[asset_idx] += 1

        if asset_defense_levels:
            least_defended_asset = min(asset_defense_levels, key=asset_defense_levels.get)
            attack_actions_for_asset = [
                i for i in range(self.num_actions)
                if i // self.config.game_setup.attacker_actions_per_asset == least_defended_asset
            ]
            return random.choice(attack_actions_for_asset)
        else:
            return random.randint(0, self.num_actions - 1)


class EnhancedTrainControlEnvironment:
    """
    增强版环境，包含智能攻击者和全面的状态建模。
    """
    def __init__(self, config):
        self.config = config
        self.gs = config.game_setup
        self.attacker = IntelligentAttacker(config)
        self.state = np.zeros(self.gs.num_defender_actions)
        self.system_health = 1.0
        self.recent_attacks = []
        self.performance_metrics = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'false_alarms': 0,
            'detection_rate': 0.0
        }
        self.defender_action_history = []
        # ======================================================================
        # 解决方案：在初始化时创建 RewardCalculator 实例
        # ======================================================================
        self.reward_calculator = RewardCalculator(config)


    def reset(self):
        self.state = np.zeros(self.gs.num_defender_actions)
        self.state[0] = 1
        return self.state.copy()

    def step(self, defender_action_idx):
        self.defender_action_history.append(defender_action_idx)
        attacker_action_idx = self.attacker.choose_attack_strategy(
            len(self.defender_action_history), self.defender_action_history)
        is_attack = attacker_action_idx is not None
        outcome = self._determine_outcome(attacker_action_idx, defender_action_idx)

        if is_attack:
            self.attacker.observe_defender_action(defender_action_idx, outcome)

        self._update_system_metrics(outcome, is_attack)
        next_state = np.zeros_like(self.state)
        next_state[defender_action_idx] = 1
        self.state = next_state
        reward_info = self._calculate_comprehensive_reward(
            outcome, attacker_action_idx, defender_action_idx)

        return self.state.copy(), reward_info, False

    def _determine_outcome(self, attacker_action_idx, defender_action_idx):
        if attacker_action_idx is None:
            base_fa_prob = self.config.model_params.false_alarm_probability[defender_action_idx]
            stress_factor = 1.0 - self.system_health
            adjusted_fa_prob = base_fa_prob * (1 + stress_factor)
            return "FP" if random.random() < adjusted_fa_prob else "TN"
        else:
            base_detection_prob = self.config.model_params.detection_probability[
                attacker_action_idx][defender_action_idx]
            performance_factor = self.performance_metrics['detection_rate']
            adjusted_detection_prob = base_detection_prob * (0.8 + 0.4 * performance_factor)
            adjusted_detection_prob = max(0.1, min(0.95, adjusted_detection_prob))
            return "TP" if random.random() < adjusted_detection_prob else "FN"

    def _update_system_metrics(self, outcome, is_attack):
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

        total_attacks = self.performance_metrics['total_attacks']
        if total_attacks > 0:
            successful_detections = total_attacks - self.performance_metrics['successful_attacks']
            self.performance_metrics['detection_rate'] = successful_detections / total_attacks

        if is_attack:
            self.recent_attacks.append(outcome)
            if len(self.recent_attacks) > 50:
                self.recent_attacks.pop(0)

    def _calculate_comprehensive_reward(self, outcome, attacker_action_idx, defender_action_idx):
        # ======================================================================
        # 解决方案：移除错误的import语句，并使用已创建的实例
        # ======================================================================
        # from core.tcs_environment import RewardCalculator # <-- 移除此行
        # reward_calc = RewardCalculator(self.config) # <-- 移除此行

        # 使用在 __init__ 中创建的实例
        attacker_payoff, defender_payoff = self.reward_calculator.get_payoffs(
            outcome, attacker_action_idx, defender_action_idx)

        performance_bonus = self._calculate_performance_bonus(outcome)
        system_health_penalty = (1.0 - self.system_health) * -10
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
        if outcome == "TP": return 5
        elif outcome == "TN": return 1
        elif outcome == "FN": return -10
        elif outcome == "FP": return -3
        return 0

    def _calculate_adaptivity_bonus(self):
        if len(self.defender_action_history) < 20:
            return 0
        recent_actions = self.defender_action_history[-20:]
        unique_actions = len(set(recent_actions))
        max_unique = min(20, self.gs.num_defender_actions)
        diversity_ratio = unique_actions / max_unique
        return diversity_ratio * 2