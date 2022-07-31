# 데이터베이스 관리
import json
import os
import time
from datetime import datetime, timedelta
import pymysql
from .config import Config
from .logger import Logger

class Database:
    def __int__(self, logger: Logger, config: Config):
        self.logger = logger
        self.config = config
    
    def create_database(self):
        pass
    
if __name__ == "__main__":
    database = Database()
    database.create_database()
