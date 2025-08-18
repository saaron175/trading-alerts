import requests

TOKEN = "8268164684:AAGgUf_tt1MnilRJayLphEv5YLvbpnF453M"
CHAT_ID = "8437423038"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload)
