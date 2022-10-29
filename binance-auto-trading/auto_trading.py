import time
from .database import Database
from .logger import Logger
from .scheduler import SafeScheduler
from .config import Config
from .api_manager import ApiManager
from .scheduler import SafeScheduler
from .test import test

def main():
    logger = Logger()
    logger.info("Starting")

    config = Config()
    db = Database(logger, config)
    manager = ApiManager(config, db, logger)

    try:
        pass
    except Exception as e:
        logger.error("Can't access Binance API")
        logger.error(e)
        return

    schedule = SafeScheduler(logger)
    schedule.every(config.PREDICTION_WAITING_TIME).seconds.do(test.testing).tag("test stage...")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    finally:
        manager.stream_manager.close()