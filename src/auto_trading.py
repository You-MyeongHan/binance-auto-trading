from .database import Database
from .logger import Logger
from .scheduler import SafeScheduler
from .config import Config
from api_manager import ApiManager
from .scheduler import SafeScheduler

def main():
    logger = Logger()
    logger.info("Starting")

    config = Config()
    db = Database(logger, config)
    manager = ApiManager(config, db, logger)

    try:
        pass
    except Exception as e:
        logger.error("Can't access Binance API - API keys may be wrong or lack sufficient permissions") # modify later
        logger.error(e)
        return

    schedule = SafeScheduler(logger)