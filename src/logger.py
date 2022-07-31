# 프로그램 구동 중 발생하는 이벤트 로그를 띄움
import logging.handlers

class Logger:
    def __init__(self):
        pass
    
    def log(self, message, level="info"):
    
        if level == "info":
            self.Logger.info(message)
        elif level == "warning":
            self.Logger.warning(message)
        elif level == "error":
            self.Logger.error(message)
            
    def info(self, message):
        self.log(message, "info")

    def warning(self, message):
        self.log(message, "warning")

    def error(self, message):
        self.log(message, "error")