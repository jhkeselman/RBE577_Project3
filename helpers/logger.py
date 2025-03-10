import logging
import sys
import threading
import os
from datetime import datetime as dt
from pathlib import Path
import time
from tensorboardX import SummaryWriter
import torch


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    blue = "\x1b[34;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    format_str = "%(asctime)s - %(levelname)s - [PID:%(process)d] [%(threadName)s] [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class ProjectLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            with self._lock:
                if not self.initialized:
                    self.logger = None
                    self.log_file = os.environ.get("PROJECT_LOG_FILE")
                    if not self.log_file:
                        self.log_file = (
                            f"logs/app_{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
                        )
                        os.environ["PROJECT_LOG_FILE"] = self.log_file
                    self.initialized = True

    def setup(self, log_level=logging.DEBUG):
        with self._lock:
            if self.logger is not None:
                return self.logger

            Path("logs").mkdir(exist_ok=True)

            logger = logging.getLogger(
                f"project_logger.{os.getpid()}.{threading.current_thread().name}"
            )
            logger.setLevel(log_level)
            logger.handlers = []

            file_handler = logging.FileHandler(self.log_file)
            console_handler = logging.StreamHandler()

            formatter = CustomFormatter()

            for handler in [file_handler, console_handler]:
                handler.setFormatter(formatter)
                handler.setLevel(log_level)
                logger.addHandler(handler)

            self.logger = logger
            return self.logger

    @property
    def get_logger(self):
        if self.logger is None:
            return self.setup()
        return self.logger

    @property
    def get_log_file(self):
        return self.log_file


project_logger = ProjectLogger()


def get_logger():
    logger = logging.getLogger("HealthcareAssistant")
    logger.setLevel(logging.DEBUG)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())

    # File Handler
    fh = logging.FileHandler("healthcare_assistant.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(CustomFormatter())

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


class Logger:
    def __init__(self, config, algorithm):
        """
        Initialize logger with TensorBoard support.

        Args:
            config (dict): Configuration dictionary
            algorithm (str): Algorithm name
        """
        self.config = config
        self.algorithm = algorithm

        # Create log directory with timestamp
        timestamp = dt.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join("logs", f"{algorithm}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Create model directory
        self.model_dir = config["logging"]["model_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Initialize ProjectLogger
        self.project_logger = project_logger.setup()

        # Training start time
        self.start_time = time.time()

        self.project_logger.info(f"Initializing {algorithm} logger")
        self.project_logger.info(f"Logging to {self.log_dir}")
        print(f"Logging to {self.log_dir}")

    def log_episode(self, episode, reward, loss=None, worker_id=None, avg_reward=None):
        """Log episode information."""
        message = (
            f"Episode {episode}/{self.config['hyperparameters']['max_episodes']} | "
        )

        if worker_id is not None:
            message += f"Worker: {worker_id} | "

        message += f"Reward: {reward:.2f}"

        if avg_reward is not None:
            message += f" | Avg Reward: {avg_reward:.2f}"

        if loss is not None:
            message += f" | Loss: {loss:.4f}"

        # Log to console and file via ProjectLogger
        self.project_logger.info(message)

        # Log to TensorBoard
        self.writer.add_scalar("Reward/episode", reward, episode)
        if avg_reward is not None:
            self.writer.add_scalar("Reward/average", avg_reward, episode)
        if loss is not None:
            self.writer.add_scalar("Loss/episode", loss, episode)

    def log_evaluation(self, episode, avg_reward):
        """Log evaluation results."""
        message = f"Evaluation at episode {episode}: Average Reward {avg_reward:.2f}"
        self.project_logger.info(message)
        self.writer.add_scalar("Reward/eval", avg_reward, episode)

    def save_model(self, episode, model_data, is_best=False):
        """Save model checkpoint."""
        if is_best:
            model_path = os.path.join(
                self.model_dir, f"{self.algorithm.lower()}_best_model.pth"
            )
            message = f"New best model saved with reward {model_data.get('best_reward', 0):.2f}"
        else:
            model_path = os.path.join(
                self.model_dir, f"{self.algorithm.lower()}_checkpoint_{episode}.pth"
            )
            message = f"Checkpoint saved to {model_path}"

        self.project_logger.info(message)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model_data, model_path)

    def close(self):
        """Close the logger."""
        training_time = time.time() - self.start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        message = f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s"
        self.project_logger.info(message)
        self.writer.close()
