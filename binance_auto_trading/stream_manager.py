import sys

import time
import binance.client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from .config import Config
from .logger import Logger

class BinanceOrder:
    def __init__(self, report):
        self.event = report
        self.symbol = report["symbol"]
        self.side = report["side"]
        self.order_type = report["order_type"]
        self.id = report["order_id"]
        self.cumulative_quote_qty = float(report["cumulative_quote_asset_transacted_quantity"])
        self.status = report["current_order_status"]
        self.price = float(report["order_price"])
        self.time = report["transaction_time"]

    def __repr__(self):
        return f"<BinanceOrder {self.event}>"

class StreamManager:
    def __init__(self, config: Config, binance_client: binance.client.Client, logger: Logger):
        
        self.logger = logger
