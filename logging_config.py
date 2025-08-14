import logging
import logging.config
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "INFO",
        },
        "info_file_handler": {
            "class": "logging.FileHandler",
            "filename": os.path.join(LOG_DIR, "info.log"),
            "formatter": "detailed",
            "level": "INFO",
        },
        "error_file_handler": {
            "class": "logging.FileHandler",
            "filename": os.path.join(LOG_DIR, "error.log"),
            "formatter": "detailed",
            "level": "ERROR",
        },
    },
    "loggers": {
        "employee_attrition": {
            "handlers": ["console", "info_file_handler", "error_file_handler"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}
