import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM,Dropout
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ccxt
import matplotlib.pyplot as plt

class lstm_prediction:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
        self.binance=ccxt.binance()
        self.btc_ohlcv = self.binance.fetch_ohlcv("BTC/USDT",'1h')
        
        self.df = pd.DataFrame(self.btc_ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        self.df=self.df[['date', 'close']]
        self.dfLen=len(self.df)
        
        self.coinData=pd.read_csv('data/Bitstamp_BTCUSDT_1h.csv')
        self.coinData=self.coinData.loc[:,['date','close']]
        self.coinData=self.coinData.loc[::-1].reset_index(drop=True)
        self.coinDataLen=len(self.coinData)
        

    def create_one_data(self):
        train_data=self.coinData.loc[:,'close'].to_frame() 
        test_data = self.df.loc[:, 'close'].to_frame() 
        
        train_data_sc=self.scaler.fit_transform(train_data) 
        test_data_sc=self.scaler.fit_transform(test_data)
        
        train_sc_df = pd.DataFrame(train_data_sc, columns=['Scaled'], index=train_data.index)
        test_sc_df = pd.DataFrame(test_data_sc, columns=['Scaled'], index=test_data.index)
        
        for i in range(1, 11):
            train_sc_df['Scaled_{}'.format(i)]=train_sc_df ['Scaled'].shift(i) # 훈련 데이터 shape 10*(len-10)으로 변환
            test_sc_df['Scaled_{}'.format(i)]=test_sc_df ['Scaled'].shift(i)

        x_train=train_sc_df.dropna().drop('Scaled', axis=1)
        y_train=train_sc_df.dropna()[['Scaled']]
        
        x_test=test_sc_df.dropna().drop('Scaled', axis=1)
        y_test=test_sc_df.dropna()[['Scaled']]
        
        train_data_sc=self.scaler.fit_transform(train_data)
        test_data_sc=self.scaler.fit_transform(test_data)
        
        train_sc_df = pd.DataFrame(train_data_sc, columns=['Scaled'], index=train_data.index)

        K.clear_session() #학습에러 방지

        model = Sequential()
        model.add(LSTM(30,return_sequences=True, input_shape=(10, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(42,return_sequences=False))
        model.add(Dropout(0.2))

        # 예측값 1개
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        early_stop = EarlyStopping(monitor='loss', patience=5) # 학습률 낮아지면 조기 종료

        model.fit(x_train, y_train, epochs=50, batch_size=20, callbacks=[early_stop])
        
        y_pred =model.predict(x_test)
        
        y_pred = self.scaler.inverse_transform(y_pred)
        one_hour_later=(self.df['date'].iloc[-1]+3600000)/1000
        pred_result=pd.DataFrame({'date':[one_hour_later], 'price':[y_pred[-1][0]]})
        pred_result=pred_result.to_json()
        return pred_result
    
    def create_ten_data():
        pass

if __name__ == "__main__":
    data = lstm_prediction()
    data=data.create_one_data()
    print(data)