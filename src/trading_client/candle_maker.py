import plotly.graph_objects as go
import pandas as pd
from binance import Client, ThreadedWebsocketManager
from PyQt5.QtCore import pyqtSignal
import time
from os import path


class CandleChartApiManager():
    CHOSEN_TICKER="BTCUSDT"
    binance=Client()
    dataSent = pyqtSignal(float, float, float ,float)
    
    def __init__(self, ticker, candleTime):
        self.twm = ThreadedWebsocketManager()
        self.CHOSEN_TICKER=ticker
        self.candleTime=candleTime
        
    def run(self):
        self.twm.start_kline_socket(symbol=self.CHOSEN_TICKER)
        ts = self.twm.symbol_book_ticker_socket(self.CHOSEN_TICKER+"@kline_"+self.candleTime)
        
        data=ts.get()
        self.dataSent.emit(float (data['content']['closePrice']),
            float(data['content']['chgRate']),float(data['content']['volumePower']))
        time.sleep(0.5)
        self.dataSent.emit(data)
            
    def changeTicker(self, ticker):
        self.CHOSEN_TICKER=ticker
        self.binance=Client.get_symbol_ticker(self.CHOSEN_TICKER)
        
    def close(self):
        self.twm.stop()