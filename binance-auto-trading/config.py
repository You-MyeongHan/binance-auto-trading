# setting file Managing file
import configparser
import os

USER_CFG="user.cfg"
USER_CFG_SECTION = "user_config"

class Config:
    def __init__(self):
        config=configparser.ConfigParser()
        config['DEFAULT']= {
            "used_coin":"BTC/USDT",
            "use_margin": "no",
            "buy_timeout":"0",
            "sell_timeout":"0",
            "strategy": "default",
            "predicion_waiting_time":"300",
            "tld": "com"
        }
        
        if not os.path.exists(USER_CFG):
            print("There is no user.cfg file. Check it out!")
            config[USER_CFG_SECTION] = {}
        else:
            config.read(USER_CFG)

        # setting api keys
        self.BINANCE_API_KEY = os.environ.get("api_key") or config.get(USER_CFG_SECTION, "api_key")
        self.BINANCE_API_SECRET_KEY = os.environ.get("api_secret_key") or config.get(USER_CFG_SECTION, "api_secret_key")

        # setting selling, buying time
        self.SELL_TIMEOUT = os.environ.get("SELL_TIMEOUT") or config.get(USER_CFG_SECTION, "sell_timeout")
        self.BUY_TIMEOUT = os.environ.get("BUY_TIMEOUT") or config.get(USER_CFG_SECTION, "buy_timeout")

        # seetting using margin or not
        self.USE_MARGIN = os.environ.get("USE_MARGIN") or config.get(USER_CFG_SECTION, "use_margin")

        self.PREDICTION_WAITING_TIME=int(os.environ.get("PREDICTION_WAITING_TIME") or config.get(USER_CFG_SECTION, "predicion_waiting_time"))