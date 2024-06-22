# run.py
#! /usr/bin/python3

import subprocess
import os

def run_fish_iot_script():
    try:
        # Set the working directory to the directory containing fish_Iot.py
        script_directory = "/home/pi/Desktop/Fish_IoT"
        os.chdir(script_directory)

        # Run fish_Iot.py script
        subprocess.run(["python", "fish_IoT.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running fish_Iot.py: {e}")

if __name__ == "__main__":
    run_fish_iot_script()
