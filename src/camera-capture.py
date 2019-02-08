from picamera import PiCamera
from time import sleep
from .lcd import i2c_device, lcd
class CaptureSession:
	def __init__(self, time):
		camera = PiCamera()
		camera.start_preview()
		sleep(time)
		camera.capture('/home/pi/Desktop/skinCAM/input/distributable-input/image.jpg')
		camera.stop_preview()
		addr = input("ADDRESS: ")
		i = lcd()
		i.lcd_display_string("Hello")
