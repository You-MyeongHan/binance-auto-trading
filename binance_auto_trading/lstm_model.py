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

class lstm_prediction:
    def __init__(self, split=0.7):
        self.scaler = MinMaxScaler()
        
        self.binance=ccxt.binance()
        self.btc_ohlcv = self.binance.fetch_ohlcv("BTC/USDT",'1h')
        
        self.df = pd.DataFrame(self.btc_ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        self.df=self.df[['time', 'close']]
        self.dfLen=len(self.df)
        
        self.coinData=pd.read_csv('data/Bitstamp_BTCUSDT_1h.csv')
        self.coinData=self.coinData.loc[:,['date','close']]
        self.coinDataLen=len(self.coinData)
        
        self.split=int(split*self.dfLen)
        self.coinSplit=int(split*self.coinDataLen)
        
        self.df['time'] = pd.to_datetime(self.df['time'], unit='ms')

    def create_one_data(self):
        train_data=self.coinData.loc[:self.coinSplit,'close'].to_frame() 
        test_data = self.coinData.loc[self.coinSplit:, 'close'].to_frame() 
        train_data_sc=self.scaler.fit_transform(train_data) 
        test_data_sc=self.scaler.fit_transform(test_data) 
        train_sc_df = pd.DataFrame(train_data_sc, columns=['Scaled'], index=train_data.index)
        test_sc_df = pd.DataFrame(test_data_sc, columns=['Scaled'], index=test_data.index)
        
        for i in range(1, 5):
            train_sc_df['Scaled_{}'.format(i)]=train_sc_df ['Scaled'].shift(i) # 훈련 데이터 형식 4*len으로 변환
            test_sc_df['Scaled_{}'.format(i)]=test_sc_df ['Scaled'].shift(i) # 테스트 데이터 형식 4*len으로 변환

        x_train=train_sc_df.dropna().drop('Scaled', axis=1)
        y_train=train_sc_df.dropna()[['Scaled']]

        x_test=test_sc_df.dropna().drop('Scaled', axis=1)
        y_test=test_sc_df.dropna()[['Scaled']]

        train_data_sc=self.scaler.fit_transform(train_data)
        test_data_sc= self.scaler.transform(test_data)

        train_sc_df = pd.DataFrame(train_data_sc, columns=['Scaled'], index=train_data.index)
        test_sc_df = pd.DataFrame(test_data_sc, columns=['Scaled'], index=test_data.index)

        K.clear_session() #학습에러 방지

        model = Sequential()
        model.add(LSTM(30,return_sequences=True, input_shape=(4, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(42,return_sequences=False))
        model.add(Dropout(0.2))

        # 예측값 1개
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        early_stop = EarlyStopping(monitor='loss', patience=5) # 학습률 낮아지면 조기 종료

        model.fit(x_train, y_train, epochs=50, batch_size=20, callbacks=[early_stop])

        y_pred = model.predict(x_test)
        y_pred = self.scaler.inverse_transform(y_pred)
        return y_pred

if __name__ == "__main__":
    data = lstm_prediction()
    data=data.create_one_data()
    