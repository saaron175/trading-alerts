import requests

TOKEN = 8268164684:AAGgUf_tt1MnilRJayLphEv5YLvbpnF453M  # paste your bot token here
CHAT_ID = 8268164684  # paste your chat ID here

def send_message(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    requests.post(url, json=payload)

# Test it:
send_message("Hello! This is a test message from my bot.")
