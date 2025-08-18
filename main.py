import time
from telegram_bot import send_telegram_message

def check_trading_signals():
    """
    Replace this logic with your real RSI/EMA or strategy code.
    For now, we'll simulate signals every 30 seconds.
    """
    import random
    signals = ["BUY", "SELL", "HOLD"]
    return random.choice(signals)

def main():
    while True:
        signal = check_trading_signals()
        print("Signal detected:", signal)
        
        if signal == "BUY":
            send_telegram_message("BUY signal detected! 🚀")
        elif signal == "SELL":
            send_telegram_message("SELL signal detected! ⚠️")
        else:
            print("No trade action right now.")
        
        time.sleep(30)  # check every 30 seconds

if __name__ == "__main__":
    main()
