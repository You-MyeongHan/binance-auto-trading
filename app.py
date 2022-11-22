import os
import re
import time
from flask import Flask, jsonify, request, current_app
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy

from binance_auto_trading.logger import *
from binance_auto_trading.config import *
from binance_auto_trading.database import *
from binance_auto_trading.api_manager import *
from binance_auto_trading.prediction import *
from binance_auto_trading.lstm_model import lstm_prediction
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] ="sqlite:///data/crypto_trading.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
db.init_app(app)

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

logger = Logger("api_server")
config = Config()
prediction=Prediction()

class User(db.Model):
    __tablename__="users"
    id=db.Column(db.String, primary_key=True)
    password=db.Column(db.String, nullable=False)
    api_key=db.Column(db.String)
    sec_key=db.Column(db.String)
    
db.create_all()

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

@app.route("/api/login",methods=['POST'])
def login():
    id=request.form['id']
    password=request.form['password']
    data={'ok':'ok'}
    return jsonify(data)

if __name__ == "__main__":
    socketio.run(app)