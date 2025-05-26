from twilio.rest import Client
from dotenv import load_dotenv
import os

from rag.index import user_input

load_dotenv()

twilio_account_sid = os.environ['TWILIO_ACCOUNT_SID']
twilio_auth_token = os.environ['TWILIO_AUTH_TOKEN']
twilio_number = os.environ['TWILIO_NUMBER']

client = Client(twilio_account_sid, twilio_auth_token)


#AIzaSyDAe2nDuiHdNZQzq-saNyI6dBeMpvRS5c0 GEMINI KEY TEST


def chat_with_phil(info, to, describe=False):
    if describe:
        client.messages.create(to=to, from_=f"whatsapp:{twilio_number}", body=info)
    else:
        msg = user_input(info)
        client.messages.create(to=to, from_=f"whatsapp:{twilio_number}", body=msg)


def send_sms(info, _from, to):
    client.messages.create(to=to, from_=f"{twilio_number}", body=info)
