# 설정 파일 관리
import configparser
import os
from .config import config

USER_CFG="user.cfg"

class Config:
    def __init__(self):
        config=configparser.ConfigParser()
        config['DEFAULT']={
            "used_coin":"BTC/USDT",
            "buy_timeout":"0",
            "sell_timeout":"0"
        }
        
        if not os.path.exists(USER_CFG):
            print("There is no user.cfg file. Check it out!")
        else:
            config.read(USER_CFG)