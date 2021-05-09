from gpiozero import Button, LED
from picamera import PiCamera
from time import sleep

from lobe import ImageModel
 
button = Button(21)
 
yellow_led = LED(17) 
blue_led = LED(27) 
green_led = LED(22) 
red_led = LED(23) 
pink_led = LED(24) 
white_led = LED(10) 
 
camera = PiCamera()

model = ImageModel.load('model/')


# Take Photo
def take_photo():
    sleep(2)
    print("Starting classify process")
    camera.start_preview(alpha=200)
    camera.rotation = 270
    camera.capture('images/predict.png')
    camera.stop_preview()
    sleep(1)

def ledOff():
        yellow_led.off()
        blue_led.off()
        green_led.off()
        red_led.off()
        white_led.off()
        pink_led.off()
 
def ledSelect(label):
    print(label)
    if label == "plastic":
        yellow_led.on()
        sleep(5)
    if label == "cardboard":
        blue_led.on()
        sleep(5)
    if label == "glass":
        red_led.on()
        sleep(5)
    if label == "paper":
        green_led.on()
        sleep(5)
    if label == "metal":
        pink_led.on()
        sleep(5)
    if label == "human":
        white_led.on()
        sleep(5)
    else:
        ledOff()



take_photo()
result = model.predict_from_file('images/predict.png')
ledSelect(f"My prediction is: {result.prediction}")
print(f"All prediction are:")
for label, confidence in result.labels:
    print(f"{label}: {confidence*100}%")




# pip3 install gpiozero
# pip3 install PiCamera
#pip3 install tensorflow --no-cache-dir (--no-cache-dir voor raspi met weinig ram)