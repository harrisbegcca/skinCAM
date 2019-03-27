Set cmd = CreateObject("WScript.shell")
Set cmd2 = CreateObject("WScript.shell")
cmd.run"cmd"
WScript.sleep 200
cmd.Sendkeys"cd  C:\Users\legos\AppData\Local\Programs\Python\Python36{Enter}"
WScript.sleep 200
cmd.Sendkeys"python C:\Users\legos\OneDrive\Desktop\SkinCAM\src\trainer-without-the-training.py{Enter}"
cmd2.run"cmd"
WScript.sleep 200
cmd2.Sendkeys"python C:\Users\legos\OneDrive\Desktop\SkinCAM\utils\rest.py{Enter}"
WScript.sleep 200
