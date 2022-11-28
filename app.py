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

app.config["SQLALCHEMY_DATABASE_URI"] ="sqlite:///data/userData.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app_ctx = app.app_context()
app_ctx.push()
db = SQLAlchemy(app)
db.init_app(app)

cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

logger = Logger("api_server")
config = Config()
prediction=Prediction()

class User(db.Model):
    __tablename__="user"
    id=db.Column(db.String, primary_key=True)
    password=db.Column(db.String, nullable=False)
    api_key=db.Column(db.String)
    sec_key=db.Column(db.String)
    
    def __repr__(self):
        return '<User %r>' % self.username

    # def check_password(self, password):
    #     if password == self.password:
    #         return True
    #     else:
    #         return False    

# db.create_all()

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
    # user=User()
    # user.id=request.form['id']
    # user.password=request.form['password']
    # db.session.add(user)
    # db.session.commit()
    
    # User.query.filter_by(id=request.form['id']).first()
    
    # try:
    #     data=User.check_password(user.password)
    #     if data is not None:
    #         pass
    # except:
    #     return "Don't login"
    
    
    data={'ok':'ok'}
    return jsonify(data)

@app.route("/api/register",methods=['POST'])
def register():
    id=request.form['id']
    password=request.form['password']

if __name__ == "__main__":
    socketio.run(app)