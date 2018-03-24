import pyrebase
import csv
import os
import firebase_admin
from firebase_admin import credentials
import urllib

cred = credentials.Certificate("C:\Users\legos\OneDrive\Desktop\Work\serviceAccountKey.json")
firebase_admin.initialize_app(cred)



firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db = firebase.database()


def stream_handler(message):
	try:
		print(message["path"]) # /-K7yGTTEp7O549EzTYtI
		print(message["data"]) # {'title': 'Pyrebase', "body": "etc..."}
		
		imgurl = "images/%s.jpg"  % message["data"]
		s = storage.child(imgurl)
		urllib.urlretrieve(s.get_url(1).split(".jpg")[0], "C:/Users/legos/Downloads/%s.jpg" % message["data"])

	except:
		print("Failed downloading")


my_stream = db.child("images").stream(stream_handler)