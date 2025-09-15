import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import pickle


def save_enhanced_results(results_df, convergence_metrics, strategy_evolution, q_table, config, logger):
    """
    Enhanced results saving with comprehensive data preservation.
    """
    try:
        # 1. 创建带时间戳的结果目录
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join("results", f"fsp_run_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"Enhanced results will be saved in: {results_dir}")

        # 2. 保存主要性能数据
        csv_path = os.path.join(results_dir, "performance_data.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Performance data saved to {csv_path}")

        # 3. 保存Q表
        q_table_path = os.path.join(results_dir, "q_table.npy")
        np.save(q_table_path, q_table)
        logger.info(f"Q-table saved to {q_table_path}")

        # 4. 保存收敛指标
        if convergence_metrics:
            convergence_path = os.path.join(results_dir, "convergence_metrics.json")
            with open(convergence_path, 'w') as f:
                json.dump(convergence_metrics, f, indent=2, default=str)
            logger.info(f"Convergence metrics saved to {convergence_path}")

        # 5. 保存策略演化数据
        if strategy_evolution:
            strategy_path = os.path.join(results_dir, "strategy_evolution.pkl")
            with open(strategy_path, 'wb') as f:
                pickle.dump(strategy_evolution, f)
            logger.info(f"Strategy evolution data saved to {strategy_path}")

        # 6. 保存配置
        config_path = os.path.join(results_dir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2, default=lambda x: x.__dict__)
        logger.info(f"Experiment configuration saved to {config_path}")

        # 7. 生成元数据
        metadata = {
            "experiment_timestamp": timestamp,
            "total_episodes": results_df['episode'].nunique() if not results_df.empty else 0,
            "total_steps": len(results_df),
            "final_performance": results_df.tail(1000)['total_reward'].mean() if len(results_df) > 1000 else 0,
            "convergence_achieved": len(convergence_metrics) > 0,
            "config_hash": hash(str(sorted(config.__dict__.items()))),
        }

        metadata_path = os.path.join(results_dir, "experiment_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Experiment metadata saved to {metadata_path}")

        return results_dir

    except Exception as e:
        logger.error(f"Failed to save enhanced results. Error: {e}")
        raise


def load_experimental_results(results_dir):
    """
    Load previously saved experimental results for analysis.
    """
    results = {}

    # Load performance data
    csv_path = os.path.join(results_dir, "performance_data.csv")
    if os.path.exists(csv_path):
        results['performance_data'] = pd.read_csv(csv_path)

    # Load Q-table
    q_table_path = os.path.join(results_dir, "q_table.npy")
    if os.path.exists(q_table_path):
        results['q_table'] = np.load(q_table_path)

    # Load convergence metrics
    convergence_path = os.path.join(results_dir, "convergence_metrics.json")
    if os.path.exists(convergence_path):
        with open(convergence_path, 'r') as f:
            results['convergence_metrics'] = json.load(f)

    # Load strategy evolution
    strategy_path = os.path.join(results_dir, "strategy_evolution.pkl")
    if os.path.exists(strategy_path):
        with open(strategy_path, 'rb') as f:
            results['strategy_evolution'] = pickle.load(f)

    # Load metadata
    metadata_path = os.path.join(results_dir, "experiment_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            results['metadata'] = json.load(f)

    return results