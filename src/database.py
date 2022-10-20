# 데이터베이스 관리
import json
import os
import time
from datetime import datetime, timedelta
#import pymysql
from .config import Config
from .logger import Logger
from .models import *

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker, scoped_session

class Database:
    def __int__(self, logger: Logger, config: Config, uri="mysql+mysqlconnector://root:2142@localhost:3306"):
        self.logger = logger
        self.config = config
        self.engine = create_engine(uri)
        self.SessionMaker = sessionmaker(bind=self.engine)


    def create_database(self):
        Base.metadata.create_all(self.engine)
    
if __name__ == "__main__":
    database = Database()
    database.create_database()
