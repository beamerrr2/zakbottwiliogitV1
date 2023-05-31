from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from ZAKCHATBOT5 import main



app = Flask(__name__)

app.config['SECRET_KEY'] = '73B82B1734'
@app.route("/ZAKCHATBOT5", methods=['GET', 'POST'])
def sms_reply():
    # Get the incoming message and phone number
    incoming_msg = request.values.get('Body', '').lower()
    phone_number = request.values.get('From', '')
    
    with open('phone_number.txt', 'w') as file:
        file.write(phone_number)
    
    # Call the main function and get the response
    result = main(phone_number, incoming_msg)

    # Start our TwiML response
    resp = MessagingResponse()

    # Add a message to the response
    resp.message(result)

    return str(resp)