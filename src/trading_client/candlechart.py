from PyQt5.QtWidgets import QWidget, QGraphicsView, QGraphicsView, QGridLayout, QApplication
from PyQt5 import uic
import finplot as fplt
import pandas as pd
import numpy as np
import ccxt
import sys
import os

CHOSEN_TICKER="BTCUSDT"

class CandleChartWindow(QGraphicsView):
    def __init__(self, parent=None, ticker=CHOSEN_TICKER):
        super().__init__(parent) 
        uic.loadUi("src/trading_client/ui_resource/candlechart_window.ui",self)
        
        binance=ccxt.binance()
        view = QGraphicsView()
        grid_layout = QGridLayout(view)
        self.resize(1000, 350)
        
        btc_ohlcv = binance.fetch_ohlcv("BTC/USDT", '1d')
        df = pd.DataFrame(btc_ohlcv, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
        df.set_index('Datetime', inplace=True)

        ax0, ax1 = fplt.create_plot_widget(master=view, rows=2, init_zoom_periods=100)
        view.axs = [ax0, ax1]
        grid_layout.addWidget(ax0.ax_widget, 0, 0)
        grid_layout.addWidget(ax1.ax_widget, 1, 0)
        ax0.reset()
        ax1.reset()

        fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']], ax=ax0)
        fplt.volume_ocv(df[['Open', 'Close', 'Volume']], ax=ax1)
        fplt.refresh() 
        fplt.show(qt_exec=False)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CandleChartWindow() 
    win.show()
    app.exec_()