import os

def log_print(msg, log_path="results/run_log.txt", clear=False):
    """
    打印并记录日志到文件
    Args:
        msg: 要打印和记录的信息（字符串）
        log_path: 日志文件路径
        clear: 是否清空旧日志（首次调用时可用）
    """
    if clear and os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            pass  # 清空文件
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
