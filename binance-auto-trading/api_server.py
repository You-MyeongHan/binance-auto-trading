import re
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from .config import Config
from .database import Database
from .logger import Logger

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

logger = Logger("api_server")
config = Config()
db = Database(logger, config)

@app.route("/api/get_balance")
def get_balance():
    pass

@app.route("/api/createSynthCoin")
def createSynthCoin():
    pass

if __name__ == "__main__":
    socketio.run(app, debug=True, port=2142)
