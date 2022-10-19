import os
import configparser

CFG_FL_NAME = "user.cfg"
USER_CFG_SECTION = "binance_user_config"
config = configparser.ConfigParser()

if not os.path.exists(CFG_FL_NAME):
    print("No configuration file (user.cfg) found!")
    config[USER_CFG_SECTION] = {}
else:
    config.read(CFG_FL_NAME)
BINANCE_API_KEY = os.environ.get("API_KEY") or config.get(USER_CFG_SECTION, "api_key")
print(BINANCE_API_KEY)