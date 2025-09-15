import json
from types import SimpleNamespace
from utils.logger import setup_logger
from utils.data_manager import save_results
from core.fsp_simulator import FspSimulator
from visualization.report_generator import ReportGenerator

def main():
    """
    Main function to run the entire simulation and reporting pipeline.
    """
    # 1. 加载配置文件
    config_path = 'config/default_config.json'
    with open(config_path, 'r') as f:
        # 使用 SimpleNamespace 将嵌套的字典转换为可以通过点号访问的对象
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    # 2. 设置日志记录器
    logger = setup_logger()
    logger.info("Configuration loaded and logger initialized.")

    # 3. 初始化并运行仿真器
    simulator = FspSimulator(config, logger)
    results_df = simulator.run_simulation()

    # 4. 保存仿真结果
    # 从仿真器中获取智能体的最终Q表
    final_q_table = simulator.defender_agent.q_table
    save_results(results_df, final_q_table, logger, output_dir='results')

    # 5. 生成可视化报告
    if not results_df.empty:
        report_generator = ReportGenerator(results_df, output_dir='results')
        report_generator.generate_all_reports(logger)
    else:
        logger.warning("Results DataFrame is empty. Skipping report generation.")

    logger.info("Project execution finished.")

if __name__ == '__main__':
    main()

