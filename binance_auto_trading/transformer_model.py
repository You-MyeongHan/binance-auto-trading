import math
import time
import numpy as np 
import pandas as pd 
import torch.nn as nn
import torch
import ccxt
import matplotlib.pyplot as plt

dataLen=500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
binance=ccxt.binance()
ohlcv = binance.fetch_ohlcv("BTC/USDT", '5m', limit=dataLen)
df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df.set_index('datetime', inplace=True)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dataLen=500):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(dataLen, d_model)
        position = torch.arange(0, dataLen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class CreateSynthData():
    
    def __init__(self, 
                 data, input_window = 10, # input 데이터 입력 형식
                 output_window = 1,       # output 데이터 입력 형식
                 batch_size = 250, 
                 lr=0.0005, 
                 epochs = 50):
        self.data=data
        self.input_window=input_window
        self.output_window=output_window
        self.batch_size=batch_size
        self.lr=lr
        self.epochs=epochs
        self.model = TransAm().to(device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.train_data, self.val_data = self.get_data(data)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.test_result=None
        self.truth=None
        
    def create_inout_sequences(self, input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+self.output_window:i+tw+self.output_window]
            inout_seq.append((train_seq, train_label))
        return torch.FloatTensor(inout_seq)
    
    def get_data(self, data):
        split = 400
        train_data = data[:split]
        test_data = data[split:]
        
        train_data = train_data.cumsum()
        train_data = 2*train_data
        test_data = test_data.cumsum()
        
        train_sequence = self.create_inout_sequences(train_data, self.input_window)
        train_sequence = train_sequence[:-self.output_window]
        
        test_data = self.create_inout_sequences(test_data, self.input_window)
        test_data = test_data[:-self.output_window]
        
        return train_sequence.to(device), test_data.to(device)
    
    def get_batch(self, source, i, batch_size):
        seq_len = min(batch_size, len(source) - 1 - i)
        data = source[i:i+seq_len]    
        inputt = torch.stack(torch.stack([item[0] for item in data]).chunk(self.input_window, 1))
        target = torch.stack(torch.stack([item[1] for item in data]).chunk(self.input_window, 1))
        return inputt, target
    
    def train(self, train_data):
        self.model.train() # T평가 모드 설정
        total_loss = 0.
        start_time = time.time()

        for batch, i in enumerate(range(0, len(train_data) - 1, self.batch_size)):
            data, targets = self.get_batch(train_data, i, self.batch_size)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
            self.optimizer.step()

            total_loss += loss.item()
            log_interval = int(len(self.train_data) / self.batch_size / 5)
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.10f} | {:5.2f} ms | '
                      'loss {:5.7f}'.format(
                        self.epochs, batch, len(train_data) // self.batch_size, self.scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss))
                total_loss = 0
                start_time = time.time()
                
    def evaluate(self, eval_model, data_source):
        eval_model.eval() # 평가 모드 설정
        total_loss = 0.
        eval_batch_size = 1000
        with torch.no_grad():
            for i in range(0, len(data_source) - 1, eval_batch_size):
                data, targets = self.get_batch(data_source, i, eval_batch_size)
                output = eval_model(data)            
                total_loss += len(data[0])* self.criterion(output, targets).cpu().item()
        return total_loss / len(data_source)
    
    def model_forecast(self, model, seqence):
        model.eval() 
        total_loss = 0.
        test_result = torch.Tensor(0)
        truth = torch.Tensor(0)

        seq = np.pad(seqence, (0, 3), mode='constant', constant_values=(0, 0))
        seq = self.create_inout_sequences(seq, self.input_window)
        seq = seq[:-self.output_window].to(device)

        seq, _ = self.get_batch(seq, 0, 1)
        with torch.no_grad():
            for i in range(0, self.output_window):            
                output = model(seq[-self.output_window:])                        
                seq = torch.cat((seq, output[-1:]))

        seq = seq.cpu().view(-1).numpy()

        return seq
    
    def forecast_seq(self, model, sequences):
        start_timer = time.time()
        model.eval() 
        forecast_seq = torch.Tensor(0)    
        actual = torch.Tensor(0)
        with torch.no_grad():
            for i in range(0, len(sequences) - 1):
                data, target = self.get_batch(sequences, i, 1)
                output = model(data)
                forecast_seq = torch.cat((forecast_seq, output[-1].view(-1).cpu()), 0)
                actual = torch.cat((actual, target[-1].view(-1).cpu()), 0)
        timed = time.time()-start_timer

        print(f"{timed} sec")

        return forecast_seq, actual
    
    def training(self): # epochs만큼 트레이닝
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.train(self.train_data)

            if(epoch % self.epochs == 0):
                val_loss = self.evaluate(self.model, self.val_data)
                print('-' * 80)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss: {:5.7f}'.format(epoch, (time.time() - epoch_start_time), val_loss))
                print('-' * 80)

            else:   
                print('-' * 80)
                print('| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time)))
                print('-' * 80)

            self.scheduler.step() 
        
    def showPlot(self):
        self.test_result, self.truth = self.forecast_seq(self.model, self.val_data)
        plt.clf()
        plt.plot(self.truth, color='red', alpha=0.7)
        plt.plot(self.test_result, color='blue', linewidth=0.7)
        plt.title('Actual vs Forecast')
        plt.legend(['Actual', 'Forecast'])
        plt.xlabel('Time Steps')
        plt.show()