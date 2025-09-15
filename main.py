# main.py (修复版本)
import json
from types import SimpleNamespace
from utils.logger import setup_logger
from utils.data_manager import save_enhanced_results  # Corrected import
from core.fsp_simulator import FictitiousSelfPlaySimulator  # 使用增强版本
from visualization.report_generator import EnhancedReportGenerator
import os


def main():
    """
    Main function to run the enhanced FSP simulation and reporting pipeline.
    """
    # 1. 加载配置文件
    config_path = 'config/default_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    # 2. 设置日志记录器
    logger = setup_logger()
    logger.info("Enhanced FSP Configuration loaded and logger initialized.")

    # 3. 初始化并运行增强的FSP仿真器
    simulator = FictitiousSelfPlaySimulator(config, logger)
    results_df, convergence_metrics, strategy_evolution = simulator.run_simulation()

    # 4. 保存仿真结果
    final_q_table = simulator.defender_agent.q_table
    # Corrected function call with all required arguments
    save_enhanced_results(results_df, convergence_metrics, strategy_evolution, final_q_table, config, logger)

    # 5. 生成增强的可视化报告
    if not results_df.empty:
        report_generator = EnhancedReportGenerator(
            results_df,
            convergence_metrics=convergence_metrics,
            strategy_evolution=strategy_evolution,
            output_dir='results'
        )
        report_generator.generate_all_reports(logger)
        logger.info("Enhanced visualization reports generated successfully.")
    else:
        logger.warning("Results DataFrame is empty. Skipping report generation.")

    logger.info("Enhanced FSP project execution finished successfully.")


def run_experimental_framework():
    """
    运行完整的实验框架进行对比分析
    """
    from experiments.experimental_framework import run_comprehensive_experiments

    logger = setup_logger()
    logger.info("Starting comprehensive experimental framework...")

    results = run_comprehensive_experiments()
    logger.info("Experimental framework completed.")

    return results


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--experimental':
        # 运行完整的实验框架
        run_experimental_framework()
    else:
        # 运行单个FSP仿真
        main()