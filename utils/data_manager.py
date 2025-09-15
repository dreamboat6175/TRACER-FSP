import os
import pandas as pd
import numpy as np
from datetime import datetime

def save_results(df, q_table, logger):
    """
    Saves the simulation results DataFrame and the agent's Q-table to a new,
    timestamped directory.

    :param df: The pandas DataFrame containing the performance data.
    :param q_table: The final Q-table from the agent.
    :param logger: The logger instance for logging messages.
    """
    try:
        # 1. 创建一个带时间戳的结果目录
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join("results", f"run_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"Results will be saved in: {results_dir}")

        # 2. 保存性能数据为 CSV 文件
        csv_path = os.path.join(results_dir, "performance_data.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Performance data saved to {csv_path}")

        # 3. 保存 Q-table 为 .npy 文件
        q_table_path = os.path.join(results_dir, "q_table.npy")
        np.save(q_table_path, q_table)
        logger.info(f"Q-table saved to {q_table_path}")

    except Exception as e:
        logger.error(f"Failed to save results. Error: {e}")

