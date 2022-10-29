import math
import time

from .config import Config
from .database import Database
from .logger import Logger
from cachetools import TTLCache, cached

from binance.client import Client


class ApiManager:
    def __init__(self, config: Config, db: Database, logger: Logger) -> None:
        self.db = db
        self.logger = logger
        self.config = config

        self.binance_client = Client(
            config.BINANCE_API_KEY,
            config.BINANCE_API_SECRET_KEY 
        )
        
    def get_trade_fees(self):
        return {ticker["symbol"]: float(ticker["takerCommission"]) for ticker in self.binance_client.get_trade_fee()}

    def get_using_bnb_for_fees(self):
        return self.binance_client.get_bnb_burn_spot_margin()["spotBNBBurn"]
    
    def get_account(self):
        return self.binance_client.get_account()

    def buy_coin(self, ):
        pass

    def sell_coin(self, ):
        pass

    def retry(self, func, *args, **kwargs):
        pass