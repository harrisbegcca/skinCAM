import pyrebase, time, os, sys, config, getpass; from os import environ
if sys.version_info>(3,0,0):
	import urllib.request
else: import urllib

root_path = r"C:\Users\legos\OneDrive\Desktop\SkinCAM"

# cred = credentials.Certificate(r"C:\Users\legos\OneDrive\Desktop\Work\serviceAccountKey.json")
# firebase_admin.initialize_app(cred)


config = config.config_key
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db = firebase.database()

i = 0
if (not environ.get("SKCM_USERNAME")):
	username = input("Username: "); pw = getpass.getpass("Password: ")
	os.environ["SKCM_USERNAME"] = username; os.environ["SKCM_PW"] = pw
username = environ.get("SKCM_USERNAME");pw = environ.get("SKCM_PW")
def stream_handler(message):
	global i
	i += 1
	auth = firebase.auth()
	user = auth.sign_in_with_email_and_password(username, pw)
	if (i != 1):
		# try:
			# print(message["path"])
			# print(message["data"])

		imgurl = "images/%s.jpg" % message["data"]
		s = storage.child(imgurl)
		user = auth.refresh(user['refreshToken'])
		# retrieve the url of the website
		u = s.get_url(user['idToken'])

		if ("images%2Furl.jpg" not in u and ("images%2F0.jpg" not in u) and "https%3A%2F%2Ffirebasestorage.googleapis" not in u and "percentages" not in u):
			print(u)

			print(i)
			if i != 1:
				if sys.version_info>(3,0,0):
					urllib.request.urlretrieve(u, root_path+r"\input\distributable-input\test.jpg")#; print("Executed request under {}".format(i))
				else: urllib.urlretrieve(u, root_path+r"\input\distributable-input\test.jpg")
				# os.system("python {}\\skincam-distributable.py".format(os.getcwd()+"\\utils")); print("Executed distributable under {}".format(i))

			data = {
				"img_url": u
			}
			db.child("images").child(message["data"]).update(data)
			time.sleep(7)
			tensor_output_file = open(root_path+r"\output\classification.log", "r")
			arraypercent = {
				"percentages": str(tensor_output_file.read()).split("[")[1].split("]")[0]
			}

			db.child("images").child(message["data"]).update(arraypercent)
			# except: print("error")
	#except:
	#	print("Invalid Error")
my_stream = db.child("images").stream(stream_handler)
