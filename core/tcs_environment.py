import numpy as np
import random
import collections


class TrainControlEnvironment:
    """
    Simulates the Train Control System (TCS) environment.
    It defines the system state, executes actions from players,
    and determines the fundamental outcome of their interaction (e.g., TP, FN, FP, TN).
    """

    def __init__(self, config):
        """
        Initializes the environment using the provided configuration.
        :param config: A SimpleNamespace object containing all simulation parameters.
        """
        self.config = config
        self.gs = config.game_setup  # Shortcut for game setup parameters
        # The state is defined by the defender's current resource allocation.
        # It's a one-hot vector representing the active defense measure.
        self.state = np.zeros(self.gs.num_defender_actions)

    def step(self, attacker_action_idx, defender_action_idx):
        """
        Executes one simulation step.
        :param attacker_action_idx: The index of the action chosen by the attacker.
        :param defender_action_idx: The index of the action chosen by the defender.
        :return: A tuple containing the outcome string and the next state.
        """
        is_attack = attacker_action_idx is not None
        outcome = None

        if is_attack:
            # If there is an attack, determine if it was detected (True Positive) or missed (False Negative).
            detection_prob = self.config.model_params.detection_probability[attacker_action_idx, defender_action_idx]
            if random.random() < detection_prob:
                outcome = "TP"  # True Positive
            else:
                outcome = "FN"  # False Negative
        else:
            # If there is no attack, determine if a false alarm was generated (False Positive) or not (True Negative).
            if random.random() < self.config.model_params.false_alarm_probability[defender_action_idx]:
                outcome = "FP"  # False Positive
            else:
                outcome = "TN"  # True Negative

        # Update the state to reflect the new defense deployment.
        next_state = np.zeros_like(self.state)
        next_state[defender_action_idx] = 1
        self.state = next_state

        return outcome, self.state


class RewardCalculator:
    """
    Encapsulates the complex reward function logic as defined in the research paper.
    This includes calculating payoffs, external rewards, intrinsic rewards, and shaping rewards.
    """

    def __init__(self, config):
        self.config = config
        self.previous_risk = 0  # Stores Risk(t-1) to calculate R_ext

    def _calculate_cyber_physical_risk(self, state, attacker_action_idx=None):
        """Calculates the Cyber-Physical Risk: Risk(t) = Likelihood(t) * Impact(t)"""
        if attacker_action_idx is None:
            return 0  # No attack, no immediate risk

        defender_action_idx = np.argmax(state)

        base_likelihood = self.config.risk_params.base_attack_likelihood[attacker_action_idx]
        defense_effectiveness = self.config.risk_params.defense_effectiveness[defender_action_idx]
        current_likelihood = base_likelihood * (1 - defense_effectiveness)
        impact = self.config.risk_params.attack_impact[attacker_action_idx]

        return current_likelihood * impact

    def calculate_external_reward(self, current_state, attacker_action_idx):
        """R_ext(t) = -(Risk(t) - Risk(t-1))"""
        current_risk = self._calculate_cyber_physical_risk(current_state, attacker_action_idx)
        r_ext = -(current_risk - self.previous_risk)
        self.previous_risk = current_risk  # Update risk for the next timestep
        return r_ext

    def calculate_intrinsic_reward(self, state, action, next_state):
        """R_int(t): Simplified simulation of the Intrinsic Curiosity Module (ICM)."""
        # This is a proxy for a full neural network-based ICM.
        # It encourages exploration by rewarding actions that are chosen less frequently.
        if not hasattr(self, 'action_counts'):
            self.action_counts = collections.defaultdict(int)

        defender_action_idx = np.argmax(state)
        self.action_counts[defender_action_idx] += 1
        return 1.0 / self.action_counts[defender_action_idx]

    def _potential_function(self, state):
        """Φ(s): Potential function for reward shaping based on expert knowledge."""
        gs = self.config.game_setup
        defender_action_idx = np.argmax(state)
        # Determine which asset is being defended
        asset_idx = defender_action_idx // gs.defender_actions_per_asset
        # Determine the level of defense on that asset
        defense_level = defender_action_idx % gs.defender_actions_per_asset
        # Get the asset's criticality weight from expert knowledge
        criticality_weight = self.config.shaping_params.asset_criticality_weights[asset_idx]

        # Potential is higher for stronger defenses on more critical assets
        return criticality_weight * defense_level

    def calculate_shaping_reward(self, prev_state, current_state):
        """R_shape(st, st+1) = γ * Φ(st+1) - Φ(st)"""
        potential_current = self._potential_function(current_state)
        potential_prev = self._potential_function(prev_state)
        return self.config.simulation.discount_factor * potential_current - potential_prev

    def get_payoffs(self, outcome, attacker_action_idx, defender_action_idx):
        """Calculates the fundamental payoffs for both players based on the game outcome."""
        gs = self.config.game_setup
        ep = self.config.economic_params

        attack_cost = ep.attack_costs[attacker_action_idx] if attacker_action_idx is not None else 0
        defense_cost = ep.defense_costs[defender_action_idx]

        asset_idx = attacker_action_idx // gs.attacker_actions_per_asset if attacker_action_idx is not None else -1
        security_value = ep.asset_values[asset_idx] if asset_idx != -1 else 0

        attacker_payoff, defender_payoff = 0, 0

        if outcome == "FN":  # Missed attack
            attacker_payoff = security_value - attack_cost
            defender_payoff = -security_value - defense_cost
        elif outcome == "FP":  # False alarm
            defender_payoff = -ep.investigation_cost - defense_cost
        elif outcome == "TP":  # Detected attack
            attacker_payoff = -attack_cost
            defender_payoff = -defense_cost
        elif outcome == "TN":  # Normal operation
            defender_payoff = -defense_cost

        return attacker_payoff, defender_payoff

