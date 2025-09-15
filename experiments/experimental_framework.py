# experiments/experimental_framework.py
import numpy as np
import pandas as pd
import json
from types import SimpleNamespace
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from utils.logger import setup_logger
from core.fsp_simulator import FictitiousSelfPlaySimulator
from agents.baseline_agents import RandomAgent, FixedStrategyAgent, GreedyAgent
from visualization.report_generator import EnhancedReportGenerator


class ExperimentalFramework:
    """
    Comprehensive experimental framework for evaluating FSP performance
    against multiple baselines with statistical rigor.
    """

    def __init__(self, base_config_path='config/default_config.json'):
        self.base_config_path = base_config_path
        self.results_dir = f"experimental_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)

        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

        # Setup logger
        self.logger = setup_logger()
        self.logger.info(f"Experimental framework initialized. Results directory: {self.results_dir}")

    def create_experiment_configs(self):
        """Create different experimental configurations for comprehensive evaluation."""
        experiments = {}

        # Base FSP configuration
        experiments['fsp_base'] = self.base_config

        # Vary learning parameters
        fsp_fast_learning = json.loads(json.dumps(self.base_config.__dict__, default=lambda x: x.__dict__))
        fsp_fast_learning['agent_params']['learning_rate'] = 0.3
        fsp_fast_learning['agent_params']['exploration_decay_rate'] = 0.999
        experiments['fsp_fast_learning'] = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v
                                                              for k, v in fsp_fast_learning.items()})

        # Vary exploration parameters
        fsp_high_exploration = json.loads(json.dumps(self.base_config.__dict__, default=lambda x: x.__dict__))
        fsp_high_exploration['agent_params']['exploration_rate'] = 1.0
        fsp_high_exploration['agent_params']['min_exploration_rate'] = 0.05
        experiments['fsp_high_exploration'] = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v
                                                                 for k, v in fsp_high_exploration.items()})

        # Vary reward weights
        fsp_intrinsic_heavy = json.loads(json.dumps(self.base_config.__dict__, default=lambda x: x.__dict__))
        fsp_intrinsic_heavy['reward_weights']['beta_int'] = 0.3
        experiments['fsp_intrinsic_heavy'] = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v
                                                                for k, v in fsp_intrinsic_heavy.items()})

        # Baseline configurations (will use different agents)
        experiments['baseline_random'] = self.base_config
        experiments['baseline_fixed'] = self.base_config
        experiments['baseline_greedy'] = self.base_config

        return experiments

    def run_single_experiment(self, exp_name, config, num_runs=5):
        """Run a single experimental configuration multiple times for statistical significance."""
        self.logger.info(f"Starting experiment: {exp_name} with {num_runs} runs")

        all_results = []
        convergence_data = []
        strategy_data = []

        for run in range(num_runs):
            self.logger.info(f"  Run {run + 1}/{num_runs}")

            try:
                if exp_name.startswith('baseline_'):
                    # Run baseline experiment
                    results_df, conv_metrics, strat_evolution = self._run_baseline_experiment(
                        exp_name, config, run)
                else:
                    # Run FSP experiment
                    simulator = FictitiousSelfPlaySimulator(config, self.logger)
                    results_df, conv_metrics, strat_evolution = simulator.run_simulation()

                # Add run identifier
                results_df['run_id'] = run
                results_df['experiment'] = exp_name

                all_results.append(results_df)
                convergence_data.extend(conv_metrics)
                strategy_data.extend(strat_evolution)

            except Exception as e:
                self.logger.error(f"Error in {exp_name} run {run}: {e}")
                continue

        # Combine all runs
        combined_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

        return {
            'results': combined_results,
            'convergence': convergence_data,
            'strategies': strategy_data,
            'experiment_name': exp_name
        }

    def _run_baseline_experiment(self, baseline_type, config, run_id):
        """Run baseline experiments with different agent types."""
        from core.enhanced_tcs_environment import EnhancedTrainControlEnvironment
        from core.performance_monitor import PerformanceMonitor

        # Create baseline agent
        if baseline_type == 'baseline_random':
            agent = RandomAgent('defender', config.game_setup.num_defender_actions)
        elif baseline_type == 'baseline_fixed':
            agent = FixedStrategyAgent('defender', config.game_setup.num_defender_actions,
                                       strategy_id=run_id % config.game_setup.num_defender_actions)
        elif baseline_type == 'baseline_greedy':
            agent = GreedyAgent('defender', config.game_setup.num_defender_actions, config)
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

        # Run simulation with baseline agent
        env = EnhancedTrainControlEnvironment(config)
        monitor = PerformanceMonitor()

        for episode in range(config.simulation.num_episodes):
            current_state = env.reset()

            for step in range(config.simulation.max_steps_per_episode):
                # Baseline agent action selection
                defender_action = agent.choose_action(current_state)

                # Environment step
                next_state, reward_info, done = env.step(defender_action)

                # Simple learning for greedy agent
                if hasattr(agent, 'learn'):
                    agent.learn(current_state, defender_action, reward_info['total_reward'], next_state)

                # Record data
                monitor.record(
                    episode=episode,
                    step=step,
                    defender_action=defender_action,
                    attacker_action=reward_info.get('attacker_action'),
                    outcome=reward_info.get('outcome'),
                    total_reward=reward_info.get('total_reward', 0),
                    system_risk=reward_info.get('detection_rate', 0),
                    epsilon=getattr(agent, 'epsilon', 0)
                )

                current_state = next_state

                if done:
                    break

        results_df = monitor.get_dataframe()
        return results_df, [], []  # No convergence or strategy data for baselines

    def run_comparative_analysis(self):
        """Run comprehensive comparative analysis across all experimental configurations."""
        self.logger.info("Starting comprehensive comparative analysis")

        # Create experiment configurations
        experiments = self.create_experiment_configs()

        # Store all experimental results
        all_experiment_results = {}

        # Run experiments
        for exp_name, config in experiments.items():
            experiment_results = self.run_single_experiment(exp_name, config, num_runs=3)
            all_experiment_results[exp_name] = experiment_results

            # Save individual experiment results
            exp_dir = os.path.join(self.results_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            if not experiment_results['results'].empty:
                experiment_results['results'].to_csv(
                    os.path.join(exp_dir, f'{exp_name}_results.csv'), index=False)

        # Generate comparative analysis
        self._generate_comparative_analysis(all_experiment_results)

        self.logger.info("Comparative analysis completed")
        return all_experiment_results

    def _generate_comparative_analysis(self, all_results):
        """Generate comprehensive comparative analysis and visualizations."""
        self.logger.info("Generating comparative analysis reports")

        # Combine results for comparison
        combined_data = []
        for exp_name, exp_data in all_results.items():
            if not exp_data['results'].empty:
                df = exp_data['results'].copy()
                df['experiment_type'] = exp_name
                combined_data.append(df)

        if not combined_data:
            self.logger.warning("No experimental data to analyze")
            return

        combined_df = pd.concat(combined_data, ignore_index=True)

        # Statistical comparison
        self._perform_statistical_analysis(combined_df)

        # Generate comparison visualizations
        self._generate_comparison_plots(all_results, combined_df)

        # Generate research paper ready results
        self._generate_paper_ready_results(combined_df)

    def _perform_statistical_analysis(self, combined_df):
        """Perform rigorous statistical analysis for research paper."""
        from scipy import stats

        analysis_path = os.path.join(self.results_dir, 'statistical_analysis.txt')

        with open(analysis_path, 'w') as f:
            f.write("COMPREHENSIVE STATISTICAL ANALYSIS\n")
            f.write("=" * 50 + "\n\n")

            # Performance comparison
            f.write("1. PERFORMANCE COMPARISON\n")
            f.write("-" * 30 + "\n")

            # Calculate final performance for each experiment
            performance_by_experiment = {}
            for exp_type in combined_df['experiment_type'].unique():
                exp_data = combined_df[combined_df['experiment_type'] == exp_type]
                # Use last 20% of episodes for final performance
                final_episodes = exp_data[exp_data['episode'] >= exp_data['episode'].max() * 0.8]
                final_performance = final_episodes.groupby('run_id')['total_reward'].sum()
                performance_by_experiment[exp_type] = final_performance.values

                f.write(f"\n{exp_type}:\n")
                f.write(f"  Mean: {final_performance.mean():.4f}\n")
                f.write(f"  Std:  {final_performance.std():.4f}\n")
                f.write(f"  Min:  {final_performance.min():.4f}\n")
                f.write(f"  Max:  {final_performance.max():.4f}\n")

            # Statistical significance tests
            f.write("\n2. STATISTICAL SIGNIFICANCE TESTS\n")
            f.write("-" * 30 + "\n")

            fsp_methods = [k for k in performance_by_experiment.keys() if k.startswith('fsp_')]
            baseline_methods = [k for k in performance_by_experiment.keys() if k.startswith('baseline_')]

            # FSP vs Baselines
            for fsp_method in fsp_methods:
                for baseline_method in baseline_methods:
                    if fsp_method in performance_by_experiment and baseline_method in performance_by_experiment:
                        fsp_perf = performance_by_experiment[fsp_method]
                        baseline_perf = performance_by_experiment[baseline_method]

                        # T-test
                        t_stat, p_value = stats.ttest_ind(fsp_perf, baseline_perf)

                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(fsp_perf) - 1) * np.var(fsp_perf, ddof=1) +
                                              (len(baseline_perf) - 1) * np.var(baseline_perf, ddof=1)) /
                                             (len(fsp_perf) + len(baseline_perf) - 2))
                        cohens_d = (np.mean(fsp_perf) - np.mean(baseline_perf)) / pooled_std

                        f.write(f"\n{fsp_method} vs {baseline_method}:\n")
                        f.write(f"  T-statistic: {t_stat:.4f}\n")
                        f.write(f"  P-value: {p_value:.6f}\n")
                        f.write(f"  Cohen's d: {cohens_d:.4f}\n")

                        if p_value < 0.05:
                            f.write(f"  Result: SIGNIFICANT (p < 0.05)\n")
                        else:
                            f.write(f"  Result: Not significant (p >= 0.05)\n")

    def _generate_comparison_plots(self, all_results, combined_df):
        """Generate publication-ready comparison plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.style.use('seaborn-v0_8-whitegrid')

        # 1. Performance comparison box plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate final performance for each run
        final_performance_data = []
        for exp_type in combined_df['experiment_type'].unique():
            exp_data = combined_df[combined_df['experiment_type'] == exp_type]
            for run_id in exp_data['run_id'].unique():
                run_data = exp_data[exp_data['run_id'] == run_id]
                final_episodes = run_data[run_data['episode'] >= run_data['episode'].max() * 0.8]
                total_reward = final_episodes['total_reward'].sum()
                final_performance_data.append({
                    'experiment': exp_type,
                    'final_performance': total_reward
                })

        final_perf_df = pd.DataFrame(final_performance_data)
        sns.boxplot(data=final_perf_df, x='experiment', y='final_performance', ax=ax)
        ax.set_title('Final Performance Comparison Across Methods', fontsize=14)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Cumulative Reward (Final 20% Episodes)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'performance_comparison_boxplot.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Learning curves comparison
        fig, ax = plt.subplots(figsize=(14, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(combined_df['experiment_type'].unique())))

        for i, exp_type in enumerate(combined_df['experiment_type'].unique()):
            exp_data = combined_df[combined_df['experiment_type'] == exp_type]

            # Calculate mean and std across runs
            episode_rewards = exp_data.groupby(['episode', 'run_id'])['total_reward'].sum().reset_index()
            mean_rewards = episode_rewards.groupby('episode')['total_reward'].agg(['mean', 'std'])

            # Apply smoothing
            window_size = 50
            smoothed_mean = mean_rewards['mean'].rolling(window=window_size).mean()
            smoothed_std = mean_rewards['std'].rolling(window=window_size).mean()

            # Plot with confidence intervals
            ax.plot(smoothed_mean.index, smoothed_mean,
                    label=exp_type, color=colors[i], linewidth=2)
            ax.fill_between(smoothed_mean.index,
                            smoothed_mean - smoothed_std,
                            smoothed_mean + smoothed_std,
                            alpha=0.2, color=colors[i])

        ax.set_title('Learning Curves Comparison', fontsize=14)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Episode Reward (Smoothed)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'learning_curves_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Security performance heatmap
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        security_metrics = ['TP', 'FP', 'FN', 'TN']

        for i, metric in enumerate(security_metrics):
            if i < len(axes):
                # Create heatmap data
                heatmap_data = []
                exp_names = []

                for exp_type in sorted(combined_df['experiment_type'].unique()):
                    exp_data = combined_df[combined_df['experiment_type'] == exp_type]
                    if 'outcome' in exp_data.columns:
                        metric_rate = (exp_data['outcome'] == metric).mean()
                        heatmap_data.append(metric_rate)
                        exp_names.append(exp_type.replace('_', '\n'))

                if heatmap_data:
                    # Create a simple bar plot instead of heatmap for better readability
                    bars = axes[i].bar(range(len(exp_names)), heatmap_data,
                                       color=plt.cm.viridis(np.array(heatmap_data)))
                    axes[i].set_title(f'{metric} Rate', fontsize=12)
                    axes[i].set_xticks(range(len(exp_names)))
                    axes[i].set_xticklabels(exp_names, rotation=45, ha='right')
                    axes[i].set_ylabel('Rate')

                    # Add value labels on bars
                    for bar, value in zip(bars, heatmap_data):
                        axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                                     f'{value:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'security_performance_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_paper_ready_results(self, combined_df):
        """Generate results table ready for research paper inclusion."""

        # Create summary statistics table
        summary_stats = []

        for exp_type in sorted(combined_df['experiment_type'].unique()):
            exp_data = combined_df[combined_df['experiment_type'] == exp_type]

            # Calculate key metrics
            final_episodes = exp_data[exp_data['episode'] >= exp_data['episode'].max() * 0.8]

            stats_dict = {
                'Method': exp_type.replace('_', ' ').title(),
                'Mean_Reward': final_episodes.groupby('run_id')['total_reward'].sum().mean(),
                'Std_Reward': final_episodes.groupby('run_id')['total_reward'].sum().std(),
                'Convergence_Episode': self._estimate_convergence_episode(exp_data),
            }

            # Security metrics if available
            if 'outcome' in exp_data.columns:
                for outcome in ['TP', 'FP', 'FN', 'TN']:
                    rate = (exp_data['outcome'] == outcome).mean()
                    stats_dict[f'{outcome}_Rate'] = rate

            summary_stats.append(stats_dict)

        # Save as CSV and LaTeX table
        summary_df = pd.DataFrame(summary_stats)

        # CSV for analysis
        summary_df.to_csv(os.path.join(self.results_dir, 'paper_ready_results.csv'), index=False)

        # LaTeX table for paper
        latex_table = self._create_latex_table(summary_df)
        with open(os.path.join(self.results_dir, 'results_table.tex'), 'w') as f:
            f.write(latex_table)

        self.logger.info("Paper-ready results generated")

    def _estimate_convergence_episode(self, exp_data):
        """Estimate the episode where learning converged."""
        if len(exp_data) == 0:
            return np.nan

        # Calculate rolling variance of rewards
        episode_rewards = exp_data.groupby('episode')['total_reward'].sum()
        rolling_var = episode_rewards.rolling(window=100).var()

        # Find where variance stabilizes (first time it goes below threshold and stays)
        threshold = rolling_var.quantile(0.25)  # 25th percentile as threshold

        for episode in rolling_var.index[100:]:  # Start after initial window
            if rolling_var[episode] < threshold:
                # Check if it stays below threshold for next 100 episodes
                future_window = rolling_var[episode:episode + 100]
                if len(future_window) >= 50 and (future_window < threshold * 1.5).all():
                    return episode

        return np.nan

    def _create_latex_table(self, summary_df):
        """Create a publication-ready LaTeX table."""

        # Format numeric columns
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.endswith('_Rate'):
                summary_df[col] = summary_df[col].apply(lambda x: f"{x:.3f}")
            elif 'Reward' in col:
                summary_df[col] = summary_df[col].apply(lambda x: f"{x:.2f}")
            elif 'Episode' in col:
                summary_df[col] = summary_df[col].apply(lambda x: f"{x:.0f}" if not pd.isna(x) else "N/A")

        latex_str = """\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison of Different Methods}
\\label{tab:performance_comparison}
\\begin{tabular}{l|c|c|c|c|c|c|c}
\\hline
\\textbf{Method} & \\textbf{Mean Reward} & \\textbf{Std Reward} & \\textbf{Convergence} & \\textbf{TP Rate} & \\textbf{FP Rate} & \\textbf{FN Rate} & \\textbf{TN Rate} \\\\
\\hline
"""

        for _, row in summary_df.iterrows():
            method = row['Method']
            mean_reward = row.get('Mean_Reward', 'N/A')
            std_reward = row.get('Std_Reward', 'N/A')
            convergence = row.get('Convergence_Episode', 'N/A')
            tp_rate = row.get('TP_Rate', 'N/A')
            fp_rate = row.get('FP_Rate', 'N/A')
            fn_rate = row.get('FN_Rate', 'N/A')
            tn_rate = row.get('TN_Rate', 'N/A')

            latex_str += f"{method} & {mean_reward} & {std_reward} & {convergence} & {tp_rate} & {fp_rate} & {fn_rate} & {tn_rate} \\\\\n"

        latex_str += """\\hline
\\end{tabular}
\\end{table}"""

        return latex_str


# Baseline agent implementations
class RandomAgent:
    """Random baseline agent for comparison."""

    def __init__(self, agent_id, num_actions):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.epsilon = 0.0  # For compatibility

    def choose_action(self, state):
        return np.random.choice(self.num_actions)


class FixedStrategyAgent:
    """Fixed strategy baseline agent."""

    def __init__(self, agent_id, num_actions, strategy_id=0):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.strategy_id = strategy_id % num_actions
        self.epsilon = 0.0

    def choose_action(self, state):
        return self.strategy_id


class GreedyAgent:
    """Greedy baseline agent that learns simple value estimates."""

    def __init__(self, agent_id, num_actions, config):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.action_values = np.zeros(num_actions)
        self.action_counts = np.zeros(num_actions)
        self.epsilon = 0.1

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.action_values)

    def learn(self, state, action, reward, next_state):
        self.action_counts[action] += 1
        # Simple incremental average
        self.action_values[action] += (reward - self.action_values[action]) / self.action_counts[action]


# Main execution function
def run_comprehensive_experiments():
    """Run the complete experimental framework."""
    framework = ExperimentalFramework()
    results = framework.run_comparative_analysis()

    print("Experimental framework completed successfully!")
    print(f"Results saved in: {framework.results_dir}")

    return results


if __name__ == "__main__":
    run_comprehensive_experiments()