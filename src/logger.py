# create logs about processing
import logging.handlers

class Logger:
    def __init__(self, logging_service="auto_trading"):
        self.Logger = logging.getLogger(f"{logging_service}_logger")
        self.Logger.setLevel(logging.DEBUG)
        self.Logger.propagate = False
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        fh = logging.FileHandler(f"logs/{logging_service}.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.Logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.Logger.addHandler(ch)
    
    def log(self, message, level="info"):
        if level == "info":
            self.Logger.info(message)
        elif level == "warning":
            self.Logger.warning(message)
        elif level == "error":
            self.Logger.error(message)
        elif level == "debug":
            self.Logger.debug(message)
            
    def info(self, message):
        self.log(message, "info")

    def warning(self, message):
        self.log(message, "warning")

    def error(self, message):
        self.log(message, "error")
            
    def debug(self, message):
        self.log(message, "debug")