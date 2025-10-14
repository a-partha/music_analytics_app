import logging
import os

def setup_logging(log_dir="logs", log_level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, "music_analytics.log")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )
