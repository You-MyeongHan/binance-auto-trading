import os
import numpy as np
import pandas as pd
from datetime import datetime
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
        
        self.createdData=pd.DataFrame(columns={'date','close'})

    def create_data(self):
        train_data=self.coinData.loc[:,'close'].to_frame() 
        test_data = self.df.loc[:, 'close'].to_frame() 
        
        train_data_sc=self.scaler.fit_transform(train_data) 
        test_data_sc=self.scaler.transform(test_data)
        
        train_sc_df = pd.DataFrame(train_data_sc, columns=['Scaled'], index=train_data.index)
        test_sc_df = pd.DataFrame(test_data_sc, columns=['Scaled'], index=test_data.index)
        
        for i in range(1, 11):
            train_sc_df['Scaled_{}'.format(i)]=train_sc_df ['Scaled'].shift(i) # 훈련 데이터 shape (len-10)*10으로 변환
            test_sc_df['Scaled_{}'.format(i)]=test_sc_df ['Scaled'].shift(i)

        x_train=train_sc_df.dropna().drop('Scaled', axis=1)
        y_train=train_sc_df.dropna()[['Scaled']]
        
        x_test=test_sc_df.dropna().drop('Scaled', axis=1)
        y_test=test_sc_df.dropna()[['Scaled']]
        
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

        model.fit(x_train, y_train, epochs=3, batch_size=20, callbacks=[early_stop]) #epoch 50이상으로
    
        for i in range(10):    
            y_pred =model.predict(x_test) #처음 490개
            
            y_test.loc[len(y_test)+10]=[y_pred[-1][0]]
            x_test.loc[len(x_test)+10]=[y_pred[-1][0], float(y_test.iloc[-1]),float(y_test.iloc[-2]),float(y_test.iloc[-3]),float(y_test.iloc[-4]),float(y_test.iloc[-5]),float(y_test.iloc[-6]),float(y_test.iloc[-7]),float(y_test.iloc[-8]),float(y_test.iloc[-9])]
            # one_hour_later=(self.df['date'].iloc[-1]+3600000*(i+1))
        
        y_pred = self.scaler.inverse_transform(y_pred) # 10시간 뒤 코인 가격 예상        
        
        price=y_pred.reshape(1,-1)[0]           #최종 가격
        date1=self.df['date'].to_numpy()[11:]
        for i in range(10):
            date1=np.append(date1, (date1[len(date1)-1]+3600000))  #최종 날짜
        
        date2=np.array([])
        for i in range(len(date1)):
            date2=np.append(date2,str(datetime.fromtimestamp(int(date1[i])/1000)))
            
        pred_result=pd.DataFrame({'date':date2, 'close':price})
        pred_result=pred_result.to_json()
        return pred_result
    
    # def append_data(self, date, price):
    #     self.createdData.loc[len(self.createdData)]=[date, price]

if __name__ == "__main__":
    data = lstm_prediction()
    data=data.create_data()
    print(data)