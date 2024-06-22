
import smbus2
import RPi.GPIO as GPIO
import time
import os
import subprocess
import glob
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Initialize the I2C interface
i2c = busio.I2C(board.SCL, board.SDA)
 
# Create an ADS1115 object
ads = ADS.ADS1115(i2c)
 
# Define the analog input channels
channelo = AnalogIn(ads, ADS.P1)

dtime = 1
Vout = 0
samples = 1
cathodeone = 5
cathodetwo = 6
tot = 0



GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(cathodeone, GPIO.OUT)
GPIO.output(cathodeone, GPIO.HIGH)
GPIO.setup(cathodetwo, GPIO.OUT)
GPIO.output(cathodetwo, GPIO.HIGH)


def capture_and_map_data(samples, cathodeone, cathodetwo, channelo, dtime):
    def map_sensor_value(Vout):
        original_min = 1
        original_max = 0
        target_min = 0.4
        target_max = 1.5
        mapped_value = ((Vout - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min
        mapped_value = max(target_min, min(mapped_value, target_max))
        return mapped_value

    for i in range(samples):
        GPIO.output(cathodeone, GPIO.HIGH)
        GPIO.output(cathodetwo, GPIO.LOW)
        time.sleep(dtime)
        GPIO.output(cathodeone, GPIO.LOW)
        GPIO.output(cathodetwo, GPIO.HIGH)
        time.sleep(dtime)
        
        Vout = channelo.voltage
        print("Voltage 1: ", channelo.voltage)
        
        mapped_value = map_sensor_value(Vout)
        print("Salinity Value:", mapped_value, "ppt")
        


while True:
    
  
    capture_and_map_data(samples, cathodeone, cathodetwo, channelo, dtime)
    
    
    time.sleep(2)
    
        


