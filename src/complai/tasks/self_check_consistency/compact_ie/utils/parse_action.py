import logging
import os

import configargparse


class StoreLoggingLevelAction(configargparse.Action):
    """This class converts string into logging level"""

    LEVELS = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    CHOICES = list(LEVELS.keys()) + [str(_) for _ in LEVELS.values()]

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super().__init__(option_strings, dest, help=help, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        """This function gets the key 'value' in the LEVELS, or just uses value"""
        level = StoreLoggingLevelAction.LEVELS.get(value, value)
        setattr(namespace, self.dest, level)


class CheckPathAction(configargparse.Action):
    """This class checks file path, if not exits, then create dir(file)"""

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super().__init__(option_strings, dest, help=help, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        """This function checks file path, if not exits, then create dir(file)"""
        parent_path = os.path.dirname(value)
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        setattr(namespace, self.dest, value)
