import logging
import os
from datetime import datetime

def setup_logger():
    """
    Sets up a logger that outputs to both the console and a file.
    The log file is saved in a 'logs' directory.
    """
    # 创建 'logs' 目录（如果它不存在）
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成带时间戳的日志文件名
    log_filename = datetime.now().strftime("simulation_%Y-%m-%d_%H-%M-%S.log")
    log_filepath = os.path.join(log_dir, log_filename)

    # 获取根 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # 设置最低日志级别

    # 如果 logger 已有 handlers, 先移除它们，防止重复输出
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建一个 formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. 创建文件 handler
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. 创建控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

