import os
import re
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from binance_auto_trading.logger import *
from binance_auto_trading.config import *
from binance_auto_trading.database import *
from binance_auto_trading.api_manager import *
from binance_auto_trading.prediction import *
from binance_auto_trading.lstm_model import lstm_prediction
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

logger = Logger("api_server")
config = Config()
db = Database(logger, config)
manager = ApiManager(config, db, logger)
prediction=Prediction()

@app.route('/')
def sessions():
    logger.info("Starting")
    return jsonify({'message':"home"})

@app.route("/api/predict", methods=['GET'])
def predict():
    epochs=request.args.get('epochs', default=50)
    model=request.args.get('model', default="lstm")
    loss=request.args.get('loss', default="mean_squared_error")
    activation=request.args.get('activation', default="tanh")
    prediction=lstm_prediction(epochs,model,loss,activation)
    data=prediction.create_data()
    
    return jsonify(data)

@app.route("/api/get_balance", methods=['GET'])
def get_balance():
    dataLen=request.form['dataLen']
    print(dataLen)
    return jsonify({'result':"success"})

@app.route("/api/createSynthCoin")
def createSynthCoin():
    print("checkPoint2")

if __name__ == "__main__":
    socketio.run(app)