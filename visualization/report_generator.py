import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ReportGenerator:
    """
    Generates and saves a series of plots to visualize the simulation results.
    """

    def __init__(self, results_df, output_dir='results'):
        """
        Initializes the report generator.
        :param results_df: A pandas DataFrame containing the simulation data.
        :param output_dir: The directory where plots will be saved.
        """
        self.df = results_df
        self.output_dir = output_dir

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Set a professional plot style
        sns.set_theme(style="whitegrid")

    def _save_plot(self, fig, filename):
        """Helper function to save a plot."""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_total_reward_over_time(self, rolling_window=100):
        """Plots the total reward per episode with a rolling average."""
        fig, ax = plt.subplots(figsize=(12, 6))
        episode_rewards = self.df.groupby('episode')['total_reward'].sum()
        episode_rewards.plot(ax=ax, alpha=0.3, label='Sum of Rewards per Episode')
        episode_rewards.rolling(window=rolling_window).mean().plot(
            ax=ax,
            linewidth=2,
            label=f'Rolling Mean (window={rolling_window})'
        )
        ax.set_title('Total Reward Over Episodes', fontsize=16)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend()
        self._save_plot(fig, 'total_reward_over_time.png')

    def plot_system_risk_over_time(self, rolling_window=100):
        """Plots the average system risk per episode with a rolling average."""
        fig, ax = plt.subplots(figsize=(12, 6))
        episode_risk = self.df.groupby('episode')['system_risk'].mean()
        episode_risk.plot(ax=ax, alpha=0.3, label='Mean System Risk per Episode')
        episode_risk.rolling(window=rolling_window).mean().plot(
            ax=ax,
            linewidth=2,
            label=f'Rolling Mean (window={rolling_window})'
        )
        ax.set_title('System Cyber-Physical Risk Over Episodes', fontsize=16)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average System Risk')
        ax.legend()
        self._save_plot(fig, 'system_risk_over_time.png')

    def plot_reward_components(self, rolling_window=100):
        """Plots the components of the reward function over time."""
        fig, ax = plt.subplots(figsize=(12, 6))
        reward_components = self.df.groupby('episode')[['extrinsic_reward', 'intrinsic_reward', 'shaping_reward']].sum()
        reward_components.rolling(window=rolling_window).mean().plot(ax=ax, linewidth=2)
        ax.set_title('Reward Components Over Episodes (Rolling Mean)', fontsize=16)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Sum of Reward Component')
        ax.legend(title='Reward Type')
        self._save_plot(fig, 'reward_components.png')

    def plot_epsilon_decay(self):
        """Plots the decay of the exploration rate (epsilon) over time."""
        fig, ax = plt.subplots(figsize=(12, 6))
        epsilon_trace = self.df.groupby('episode')['epsilon'].first()  # Get the epsilon at the start of each episode
        epsilon_trace.plot(ax=ax, linewidth=2)
        ax.set_title('Epsilon (Exploration Rate) Decay', fontsize=16)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon Value')
        self._save_plot(fig, 'epsilon_decay.png')

    def plot_outcome_distribution(self):
        """Plots the distribution of outcomes (TP, FP, FN, TN)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='outcome', data=self.df, ax=ax, order=['TP', 'FN', 'FP', 'TN'])
        ax.set_title('Distribution of Outcomes Over Entire Simulation', fontsize=16)
        ax.set_xlabel('Outcome Type')
        ax.set_ylabel('Count')
        # Add percentages on top of bars
        total = len(self.df)
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / total)
            x = p.get_x() + p.get_width() / 2 - 0.1
            y = p.get_height() + 5
            ax.annotate(percentage, (x, y))
        self._save_plot(fig, 'outcome_distribution.png')

    def generate_all_reports(self, logger):
        """Generates and saves all available plots."""
        logger.info(f"Generating reports in directory: {self.output_dir}")
        self.plot_total_reward_over_time()
        self.plot_system_risk_over_time()
        self.plot_reward_components()
        self.plot_epsilon_decay()
        self.plot_outcome_distribution()
        logger.info("All reports generated successfully.")

