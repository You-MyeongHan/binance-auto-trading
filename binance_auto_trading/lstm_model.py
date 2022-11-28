import os
import numpy as np
import pandas as pd
from datetime import datetime
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
    def __init__(self,epochs,model,loss,activation):
        self.epochs=int(epochs)
        self.model=model
        self.loss=loss
        self.activation=activation

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
        model.add(Dense(1, activation=self.activation))
        model.compile(loss=self.loss, optimizer='adam')
        early_stop = EarlyStopping(monitor='loss', patience=5) # 학습률 낮아지면 조기 종료

        model.fit(x_train, y_train, epochs=self.epochs, batch_size=20, callbacks=[early_stop]) #epoch 50이상으로
    
        for i in range(10):    
            y_pred =model.predict(x_test) #처음 490개
            
            y_test.loc[len(y_test)+10]=[y_pred[-1][0]]
            x_test.loc[len(x_test)+10]=[y_pred[-1][0], float(y_test.iloc[-1]),float(y_test.iloc[-2]),float(y_test.iloc[-3]),float(y_test.iloc[-4]),float(y_test.iloc[-5]),float(y_test.iloc[-6]),float(y_test.iloc[-7]),float(y_test.iloc[-8]),float(y_test.iloc[-9])]
        
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

def sigmoid(x):
    return 1/(1+np.exp(-x))

def clip_grad(grads, max_norm):
    norm=np.sqrt(np.sum(grads*grads))
    r=max_norm/norm
    if r<1:
        clipped_grads=grads*r
    else:
        clipped_grads=grads
    return clipped_grads

class LSTMLayer:
    def __init__(self,n_upper,n):   # n_upper : 앞 층의 뉴런 수, n : 해당 층의 뉴런 수
        self.w=np.random.rand(4, n_upper, n) / np.sqrt(n_upper) # 자비에르 초기화 
        self.v=np.random.rand(4, n, n) / np.sqrt(n)
        self.b=np.zeros((4,n))
    
    def forward(self, x, y_prev, c_prev):
        u=np.matmul(x, self.w)+np.matmul(y_prev, self.v)+self.b.reshape(4,1,-1)
        
        a0=sigmoid(u[0])    # forget gates
        a1=sigmoid(u[1])    # input gates
        a2=sigmoid(u[2])    # C값
        a3=sigmoid(u[3])    # output gates
        self.gates=np.stack((a0,a1,a2,a3))
        self.c=a0*c_prev+a1*a2 # 기억 셀
        self.y=a3*np.tanh(self.c) # 출력
        
    def backward(self, x, y, c, y_prev, c_prev, gates, grad_y, grad_c):
        a0, a1, a2, a3 = gates
        tanh_c=np.tanh(c)
        r=grad_c+(grad_y*a3)*(1-tanh_c**2)
        
        delta_a0=r*c_prev*a0*(1-a0)
        delta_a1=r*a2*a1*(1-a1)
        delta_a2=r*a1*(1-a2**2)
        delta_a3=grad_y*tanh_c*a3*(1-a3)
        
        deltas=np.stack((delta_a0, delta_a1, delta_a2, delta_a3))
        
        self.grad_w +=np.matmul(x.T, deltas)
        self.grad_v +=np.matmul(y_prev.T, deltas)
        self.grad_b +=np.matmul(deltas, deltas)
        
        grad_x=np.matmul(deltas, self.w.transpose(0,2,1))
        self.grad_x=np.sum(grad_x, axis=0)
        
        grad_y_prev=np.matmul(deltas, self.v.transpose(0,2,1))
        self.grad_y_prev=np.sum(grad_y_prev, axis=0)
        
        self.grad_c_prev=r*a0
        
    def reset_sum_grad(self):
        self.grad_w=np.zeros_like(self.w)
        self.grad_v=np.zeros_like(self.v)
        self.grad_b=np.zeros_like(self.b)
        
    def update(self,eta):
        self.w-=eta*self.grad_w
        self.v-=eta*self.grad_v
        self.b-=eta*self.grad_b
        
    def clip_grads(self, clip_const):
        self.grad_w=clip_grad(self.grad_w, clip_const*np.const*np.sqrt(self.grad_w.size))
        self.grad_v=clip_grad(self.grad_v, clip_const*np.const*np.sqrt(self.grad_v.size))
        
class OutputLayer:
    def __init__(self, n_upper, n):
        self.w=np.random.rand(n_upper, n) / np.sqrt(n_upper)
        self.b=np.zeros(n)
        
    def forward(self, x):
        self.x=x
        u=np.dot(x,self.w) + self.b
        self.y=np.exp(u)/np.sum(np.exp(u), axis=1).reshape(-1,1)
        
    def backward(self ,t):
        delta=self.y-t
        
        self.grad_w=np.dot(self.x.T, delta)
        self.grad_b=np.dot(delta, axis=0)
        self.grad_x=np.dot(delta, self.W.T)
        
    def update(self, eta):
        self.w-=eta*self.grad_w
        self.b-=eta*self.grad_b
        
def train(x_mb, t_mb):
    y_rnn=np.zeros((len(x_mb), n_time+1, n_mid))
    c_rnn=np.zeros((len(x_mb), n_time+1, n_mid))
    gates_rnn=np.zeros((4,len(x_mb), n_time, n_mid))
    y_prev=y_rnn[:,0,:]
    c_prev=c_rnn[:,0,:]
    
    for i in range(n_time):
        x=x_mb[:,i,:]
        lstm_layer.forward(x,y_prev, c_prev)
        
        y=lstm_layer.y
        y_rnn[:,i+1,:]=y
        y_pred=y
        
        c=lstm_layer.c
        c_rnn[:,i+1,:]=c
        c_prev=c
        
        gates=lstm_layer.gates
        gates_rnn[:,:,i,:]=gates
    
    #순전파 출력층
    output_layer.forward(y)
    #역전파 출력층
    output_layer.backward(t_mb)
    grad_y=output_layer.grad_x
    grad_c=np.zeros_like(lstm_layer.c)
    
    lstm_layer.reset_sum_grad()
    for i in reversed(range(n_time)):
        x=x_mb[:,i,:]
        y=y_rnn[:,i,:]
        c=c_rnn[:,i,:]
        gates=gates_rnn[:,:,i,:]
        
        lstm_layer.backward(x,y,c,y_prev, c_prev, gates, grad_y, grad_c)
        grad_y=lstm_layer.grad_y_prev
        grad_c=lstm_layer.grad_c_prev
        
    lstm_layer.update(lr)
    output_layer.update(lr)

def predict(x_mb):
    y_prev=np.zeros((len(x_mb),n_mid))
    c_prev=np.zeros((len(x_mb),n_mid))
    for i in range(n_time):
        x=x_mb[:,i,:]
        lstm_layer.forward(x,y_prev,c_prev)
        y=lstm_layer.y
        y_prev=y
        c=lstm_layer.c
        c_prev=c
        
    output_layer.forward(y)
    
    return output_layer.y

# MSE : 평균 제곱 오차
def get_error(x,t):
    y=predict(x)
    return 1.0/len(t)*np.sum(np.square(y-t))


if __name__ == "__main__":
    # data = lstm_prediction(epochs=10, model='LSTM', loss='MSE', activation='tanh')
    # data=data.create_data()                                
    # print(data)
    n_time=10 # 시점 수
    n_in=1 #입력층 누런 수 : n_upper
    n_mid=20 # 은닉층 뉴런 수 : n
    n_out=1
    
    lr=0.01 # 학습률
    epochs=11
    batch_size=8
    interval=3
    
    scaler = MinMaxScaler()
    coinData=pd.read_csv('data/Bitstamp_BTCUSDT_1h.csv')
    coinData=coinData.loc[:,['date','close']]
    coinData=coinData.loc[::-1].reset_index(drop=True)
    coinDataLen=len(coinData)
        
    createdData=pd.DataFrame(columns={'date','close'})
    
    train_data=coinData.loc[:,'close'].to_frame()
    
    train_data_sc=scaler.fit_transform(train_data)
    
    train_sc_df = pd.DataFrame(train_data_sc, columns=['Scaled'], index=train_data.index)

    lstm_layer=LSTMLayer(n_in, n_mid)
    output_layer=OutputLayer(n_mid, n_out)
    
    error_record=[]
    n_batch=len(train_sc_df) // batch_size
    
    for i in range(epochs):
        index_random=np.arange(len(train_sc_df))
        np.random.shuffle(index_random)

        for j in range(n_batch):
            mb_index=index_random[j*batch_size:(j+1)*batch_size]
            x_mb=train_sc_df[mb_index,:]
            t_mb=correct_data[mb_index,:]
            train(x_mb, t_mb)

        error=get_error(train_sc_df, correct_data)
        error_record.append(error)
        
        if i%interval==0:
            print("epoch:"+str(i+1)+"/"+str(epochs)+"error:"+str(error))
    
    predict
    x=np.array(predicted[])
    predict(x)
            