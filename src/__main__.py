#auto_trading부터 시작
from .auto_trading import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass