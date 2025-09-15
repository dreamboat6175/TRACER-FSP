# visualization/enhanced_report_generator.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')


class EnhancedReportGenerator:
    """
    Enhanced report generator with comprehensive analysis for FSP simulation results.
    """

    def __init__(self, results_df, convergence_metrics=None, strategy_evolution=None, output_dir='results'):
        self.df = results_df
        self.convergence_metrics = convergence_metrics or []
        self.strategy_evolution = strategy_evolution or []
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Set professional style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def _save_plot(self, fig, filename):
        """Enhanced plot saving with high DPI and metadata."""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)

    def plot_learning_convergence(self, window_size=100):
        """Analyze and plot learning convergence with statistical tests."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Reward convergence with confidence intervals
        episode_rewards = self.df.groupby('episode')['total_reward'].mean()
        rolling_mean = episode_rewards.rolling(window=window_size).mean()
        rolling_std = episode_rewards.rolling(window=window_size).std()

        ax1.plot(episode_rewards.index, episode_rewards, alpha=0.3, label='Episode Reward')
        ax1.plot(rolling_mean.index, rolling_mean, linewidth=2, label=f'Rolling Mean ({window_size})')
        ax1.fill_between(rolling_mean.index,
                         rolling_mean - rolling_std,
                         rolling_mean + rolling_std,
                         alpha=0.2, label='±1 Std')
        ax1.set_title('Learning Convergence Analysis')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Detection rate evolution
        if 'detection_rate' in self.df.columns:
            episode_detection = self.df.groupby('episode')['detection_rate'].mean()
            rolling_detection = episode_detection.rolling(window=window_size).mean()

            ax2.plot(rolling_detection.index, rolling_detection, linewidth=2, color='red')
            ax2.set_title('Detection Rate Evolution')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Detection Rate')
            ax2.grid(True, alpha=0.3)

        # 3. System health over time
        if 'system_health' in self.df.columns:
            episode_health = self.df.groupby('episode')['system_health'].mean()
            rolling_health = episode_health.rolling(window=window_size).mean()

            ax3.plot(rolling_health.index, rolling_health, linewidth=2, color='green')
            ax3.set_title('System Health Evolution')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('System Health')
            ax3.grid(True, alpha=0.3)

        # 4. Convergence rate analysis
        if len(episode_rewards) > 200:
            # Calculate convergence rate using variance reduction
            variance_windows = []
            window_centers = []
            for i in range(100, len(episode_rewards), 50):
                window_data = episode_rewards[i - 100:i].values
                variance_windows.append(np.var(window_data))
                window_centers.append(i)

            ax4.plot(window_centers, variance_windows, linewidth=2, color='purple')
            ax4.set_title('Convergence Rate (Reward Variance)')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Reward Variance')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_plot(fig, 'learning_convergence_analysis.png')

    def plot_strategy_evolution(self):
        """Plot evolution of player strategies over time."""
        if not self.strategy_evolution:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        episodes = [data['episode'] for data in self.strategy_evolution]

        # Defender strategy evolution
        defender_strategies = np.array([data['defender_distribution']
                                        for data in self.strategy_evolution])

        im1 = ax1.imshow(defender_strategies.T, aspect='auto', cmap='viridis',
                         extent=[episodes[0], episodes[-1], 0, len(defender_strategies[0])])
        ax1.set_title('Defender Strategy Evolution')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Defense Action')
        plt.colorbar(im1, ax=ax1, label='Probability')

        # Attacker strategy evolution
        attacker_strategies = np.array([data['attacker_distribution']
                                        for data in self.strategy_evolution])

        im2 = ax2.imshow(attacker_strategies.T, aspect='auto', cmap='plasma',
                         extent=[episodes[0], episodes[-1], 0, len(attacker_strategies[0])])
        ax2.set_title('Attacker Strategy Evolution')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Attack Action')
        plt.colorbar(im2, ax=ax2, label='Probability')

        plt.tight_layout()
        self._save_plot(fig, 'strategy_evolution.png')

    def plot_performance_comparison(self, baseline_results=None):
        """Compare performance against baselines with statistical significance."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Reward distribution comparison
        final_rewards = self.df[self.df['episode'] >= self.df['episode'].max() - 500]['total_reward']

        ax1.hist(final_rewards, bins=50, alpha=0.7, label='FSP Method', density=True)
        if baseline_results is not None:
            baseline_rewards = baseline_results[baseline_results['episode'] >=
                                                baseline_results['episode'].max() - 500]['total_reward']
            ax1.hist(baseline_rewards, bins=50, alpha=0.7, label='Baseline', density=True)

            # Statistical test
            t_stat, p_value = stats.ttest_ind(final_rewards, baseline_rewards)
            ax1.text(0.05, 0.95, f'T-test p-value: {p_value:.4f}',
                     transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax1.set_title('Final Performance Distribution')
        ax1.set_xlabel('Total Reward')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Cumulative performance
        cumulative_rewards = self.df.groupby('episode')['total_reward'].sum().cumsum()
        ax2.plot(cumulative_rewards.index, cumulative_rewards, linewidth=2, label='FSP Method')

        if baseline_results is not None:
            baseline_cumulative = baseline_results.groupby('episode')['total_reward'].sum().cumsum()
            ax2.plot(baseline_cumulative.index, baseline_cumulative,
                     linewidth=2, label='Baseline', linestyle='--')

        ax2.set_title('Cumulative Reward Comparison')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Security metrics
        security_metrics = self.df.groupby('episode')[['system_risk', 'detection_rate']].mean()

        ax3_twin = ax3.twinx()
        line1 = ax3.plot(security_metrics.index, security_metrics['system_risk'],
                         color='red', linewidth=2, label='System Risk')
        line2 = ax3_twin.plot(security_metrics.index, security_metrics['detection_rate'],
                              color='blue', linewidth=2, label='Detection Rate')

        ax3.set_xlabel('Episode')
        ax3.set_ylabel('System Risk', color='red')
        ax3_twin.set_ylabel('Detection Rate', color='blue')
        ax3.set_title('Security Performance Metrics')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper right')

        # 4. Action diversity analysis
        action_entropy = []
        for episode in range(0, self.df['episode'].max(), 100):
            episode_actions = self.df[self.df['episode'] == episode]['defender_action'].values
            if len(episode_actions) > 0:
                unique, counts = np.unique(episode_actions, return_counts=True)
                probs = counts / len(episode_actions)
                entropy = -np.sum(probs * np.log2(probs + 1e-8))
                action_entropy.append((episode, entropy))

        if action_entropy:
            episodes, entropies = zip(*action_entropy)
            ax4.plot(episodes, entropies, linewidth=2, marker='o')
            ax4.set_title('Strategy Diversity (Entropy)')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Action Entropy')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_plot(fig, 'performance_comparison.png')

    def plot_threat_landscape_analysis(self):
        """Analyze and visualize threat landscape and defense effectiveness."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Attack-Defense Interaction Heatmap
        if 'attacker_action' in self.df.columns and 'defender_action' in self.df.columns:
            # Filter out None attacks
            attack_defense_df = self.df[self.df['attacker_action'].notna()]

            interaction_matrix = pd.crosstab(
                attack_defense_df['attacker_action'],
                attack_defense_df['defender_action'],
                normalize='columns'
            )

            sns.heatmap(interaction_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1)
            ax1.set_title('Attack-Defense Interaction Frequency')
            ax1.set_xlabel('Defense Action')
            ax1.set_ylabel('Attack Action')

        # 2. Outcome distribution over time
        outcome_evolution = self.df.groupby('episode')['outcome'].value_counts(normalize=True).unstack(fill_value=0)

        for outcome in outcome_evolution.columns:
            rolling_mean = outcome_evolution[outcome].rolling(window=100).mean()
            ax2.plot(rolling_mean.index, rolling_mean, linewidth=2, label=outcome)

        ax2.set_title('Security Outcome Evolution')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Proportion')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Resource allocation efficiency
        if 'defender_action' in self.df.columns:
            action_rewards = self.df.groupby('defender_action')['total_reward'].agg(['mean', 'std', 'count'])

            ax3.bar(action_rewards.index, action_rewards['mean'],
                    yerr=action_rewards['std'], capsize=5, alpha=0.7)
            ax3.set_title('Defense Action Effectiveness')
            ax3.set_xlabel('Defense Action')
            ax3.set_ylabel('Average Reward')
            ax3.grid(True, alpha=0.3)

        # 4. Threat level vs Performance
        if 'threat_level' in self.df.columns:
            threat_bins = pd.cut(self.df['threat_level'], bins=5,
                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            threat_performance = self.df.groupby(threat_bins)['total_reward'].agg(['mean', 'std'])

            ax4.bar(range(len(threat_performance)), threat_performance['mean'],
                    yerr=threat_performance['std'], capsize=5, alpha=0.7)
            ax4.set_xticks(range(len(threat_performance)))
            ax4.set_xticklabels(threat_performance.index, rotation=45)
            ax4.set_title('Performance vs Threat Level')
            ax4.set_xlabel('Threat Level')
            ax4.set_ylabel('Average Reward')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_plot(fig, 'threat_landscape_analysis.png')

    def generate_statistical_report(self):
        """Generate comprehensive statistical analysis report."""
        report_path = os.path.join(self.output_dir, 'statistical_analysis.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRACER-FSP STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Basic statistics
            f.write("1. BASIC PERFORMANCE STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Episodes: {self.df['episode'].nunique()}\n")
            f.write(f"Total Steps: {len(self.df)}\n")
            f.write(f"Average Reward per Episode: {self.df.groupby('episode')['total_reward'].sum().mean():.4f}\n")
            f.write(f"Reward Standard Deviation: {self.df.groupby('episode')['total_reward'].sum().std():.4f}\n")

            # Security metrics
            if 'outcome' in self.df.columns:
                f.write(f"\n2. SECURITY PERFORMANCE METRICS\n")
                f.write("-" * 40 + "\n")
                outcome_counts = self.df['outcome'].value_counts()
                total_outcomes = len(self.df)

                for outcome, count in outcome_counts.items():
                    percentage = (count / total_outcomes) * 100
                    f.write(f"{outcome}: {count} ({percentage:.2f}%)\n")

                # Calculate key security metrics
                tp = outcome_counts.get('TP', 0)
                fp = outcome_counts.get('FP', 0)
                tn = outcome_counts.get('TN', 0)
                fn = outcome_counts.get('FN', 0)

                if (tp + fn) > 0:
                    sensitivity = tp / (tp + fn)
                    f.write(f"\nSensitivity (True Positive Rate): {sensitivity:.4f}\n")

                if (tn + fp) > 0:
                    specificity = tn / (tn + fp)
                    f.write(f"Specificity (True Negative Rate): {specificity:.4f}\n")

                if (tp + fp) > 0:
                    precision = tp / (tp + fp)
                    f.write(f"Precision: {precision:.4f}\n")

                if (tp + fn) > 0:
                    recall = tp / (tp + fn)
                    f.write(f"Recall: {recall:.4f}\n")

                if precision > 0 and recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    f.write(f"F1-Score: {f1_score:.4f}\n")

            # Convergence analysis
            if self.convergence_metrics:
                f.write(f"\n3. CONVERGENCE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                final_metrics = self.convergence_metrics[-10:]  # Last 10 measurements

                if final_metrics:
                    avg_variance = np.mean([m['reward_variance'] for m in final_metrics])
                    avg_entropy = np.mean([m['strategy_entropy'] for m in final_metrics])

                    f.write(f"Final Reward Variance: {avg_variance:.6f}\n")
                    f.write(f"Final Strategy Entropy: {avg_entropy:.4f}\n")

                    # Convergence test
                    if avg_variance < 1.0:
                        f.write("Status: CONVERGED (Low variance)\n")
                    else:
                        f.write("Status: STILL LEARNING (High variance)\n")

            # Learning efficiency
            f.write(f"\n4. LEARNING EFFICIENCY\n")
            f.write("-" * 40 + "\n")

            early_performance = self.df[self.df['episode'] < 500]['total_reward'].mean()
            late_performance = self.df[self.df['episode'] >= self.df['episode'].max() - 500]['total_reward'].mean()
            improvement = late_performance - early_performance

            f.write(f"Early Performance (Episodes 0-500): {early_performance:.4f}\n")
            f.write(f"Late Performance (Last 500 episodes): {late_performance:.4f}\n")
            f.write(f"Performance Improvement: {improvement:.4f}\n")
            f.write(f"Improvement Percentage: {(improvement / abs(early_performance)) * 100:.2f}%\n")

    def generate_all_reports(self, logger, baseline_results=None):
        """Generate comprehensive analysis reports."""
        logger.info(f"Generating enhanced reports in directory: {self.output_dir}")

        try:
            self.plot_learning_convergence()
            logger.info("✓ Learning convergence analysis completed")

            self.plot_strategy_evolution()
            logger.info("✓ Strategy evolution analysis completed")

            self.plot_performance_comparison(baseline_results)
            logger.info("✓ Performance comparison analysis completed")

            self.plot_threat_landscape_analysis()
            logger.info("✓ Threat landscape analysis completed")

            self.generate_statistical_report()
            logger.info("✓ Statistical analysis report generated")

            # Generate summary plots from original class
            self._generate_summary_plots()
            logger.info("✓ Summary plots generated")

        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            raise

        logger.info("All enhanced reports generated successfully.")

    def _generate_summary_plots(self):
        """Generate additional summary plots."""
        # Reward components over time
        fig, ax = plt.subplots(figsize=(12, 6))

        if all(col in self.df.columns for col in ['extrinsic_reward', 'intrinsic_reward']):
            reward_components = self.df.groupby('episode')[
                ['extrinsic_reward', 'intrinsic_reward', 'total_reward']].sum()

            rolling_components = reward_components.rolling(window=100).mean()
            rolling_components.plot(ax=ax, linewidth=2)

            ax.set_title('Reward Components Evolution')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)

        self._save_plot(fig, 'reward_components_evolution.png')