import logging
import os


def init_logger(
    root_log_level=logging.DEBUG,
    console_log_level=logging.NOTSET,
    log_file=None,
    log_file_level=logging.NOTSET,
):
    """This funtion initializes a customized logger

    Keyword Arguments:
        root_log_level {int} -- root logging level (default: {logging.DEBUG})
        console_log_level {int} -- console logging level (default: {logging.NOTSET})
        log_file {str} -- logging file path (default: {None})
        log_file_level {int} -- logging file level (default: {logging.NOTSET})
    """
    log_format = logging.Formatter(
        "[%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s]: %(message)s"
    )
    handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(log_format)
    handlers.append(console_handler)

    if log_file is not None and log_file != "":
        if os.path.exists(log_file):
            os.remove(log_file)
        elif not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        handlers.append(file_handler)

    logging.basicConfig(level=root_log_level, handlers=handlers)
