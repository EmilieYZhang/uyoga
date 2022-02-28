import os
from twilio.rest import Client

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)

message = client.messages.create(
                              body='Your friend has just finished a workout on UYOGA! To continue the cycle, complete yours.',
                              from_='+1', #sender's phone number,
                              to='+1' #user's phone number
                          )

print(message.sid)