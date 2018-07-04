from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

from pizza import pizaa

app = Flask(__name__)

@app.route('/sms',methods=['POST'])
def sms_reply():
	resp = MessagingResponse()

	if request.form['NumMedia'] != '0':
		image_url = request.form['MediaUrl0']
		resp.message(pizaa(image_url))
	else:
		resp.message('Please Send an Pizza image. Or contact the Developer')

	return str(resp)

if __name__ == '__main__':
	app.run()