import requests

# Replace with your actual bot token and chat ID
TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_telegram_message(message):
    """
    Sends a text message to your Telegram bot chat.
    """
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        print("Message sent:", message)
    except Exception as e:
        print("Failed to send message:", e)
