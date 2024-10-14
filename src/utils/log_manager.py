#    Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import datetime
import logging
from pathlib import Path


class LogManager:
    """Logger instantiation logic and keeps track of logger folder project-wide"""

    def __init__(self):
        self.logger_folder = None

    def set_logger_folder(self, logger_folder: Path):
        """
        Sets the folder where the logger will save the log files.

        Args:
            logger_folder (Path): The path to the folder where the log files will be saved.
        """

        self.logger_folder = logger_folder
        self.logger_folder.mkdir(parents=True, exist_ok=True)

    def get_new_logger(self, logger_name: str) -> logging.Logger:
        """
        Creates and returns a new logger with the specified name.

        Args:
            logger_name (str): The name of the logger.

        Returns:
            logging.Logger: The newly created logger.

        Raises:
            ValueError: If the name of the logger folder has not been set.

        """

        if not self.logger_folder:
            raise ValueError(
                "Name of logger folder has not been set, please use set_logger_folder first."
            )

        # Create a logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_file = self.logger_folder / f"{logger_name}--{current_time}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger
        return logger


log_manager = LogManager()
set_logger_folder = log_manager.set_logger_folder
