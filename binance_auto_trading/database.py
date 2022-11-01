# 데이터베이스 관리
import json
import os
import time
from datetime import datetime, timedelta
from .config import Config
from .logger import Logger
from .models import *
from socketio.exceptions import ConnectionError as SocketIOConnectionError
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker, scoped_session

class Database:
    def __init__(self, logger: Logger, config: Config, uri="sqlite:///data/crypto_trading.db"):     #mysql+pymysql://root:2142@localhost:3306
        self.logger = logger
        self.config = config
        self.engine = create_engine(uri)
        self.SessionMaker = sessionmaker(bind=self.engine)
    
    def socketio_connect(self):
        if self.socketio_client.connected and self.socketio_client.namespaces:
            return True
        try:
            if not self.socketio_client.connected:
                self.socketio_client.connect("http://api:2142", namespaces=["/backend"])
            while not self.socketio_client.connected or not self.socketio_client.namespaces:
                time.sleep(0.1)
            return True
        except SocketIOConnectionError:
            return False

    def create_database(self):
        Base.metadata.create_all(self.engine)

    def send_update(self, model):
        if not self.socketio_connect():
            return

        self.socketio_client.emit(
            "update",
            {"table": model.__tablename__, "data": model.info()},
            namespace="/backend",
        )
    
if __name__ == "__main__":
    database = Database()
    database.create_database()
