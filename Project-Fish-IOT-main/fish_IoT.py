import time
import pyrebase
import smbus2
import RPi.GPIO as GPIO
import os
import subprocess
import glob
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import numpy as np
from joblib import load
import tensorflow as tf
import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from tflite_runtime.interpreter import Interpreter

# These two lines mount the device:
os.system('modprobe w1-gpio')
os.system('modprobe w1-therm')

base_dir = '/sys/bus/w1/devices/'
device_path = glob.glob(base_dir + '28*')[0]  # get file path of sensor
rom = device_path.split('/')[-1]  # get rom name

# Initialize the I2C interface
i2c = busio.I2C(board.SCL, board.SDA)
# Create an ADS1115 object
ads = ADS.ADS1115(i2c)
# Define the analog input channel
channel = AnalogIn(ads, ADS.P0)
channelo = AnalogIn(ads, ADS.P1)
bh1750_address = 0x23
light_intensity = 0
time_for_upload_data = 0
time_for_download_data = 0
dtime = 0.005
Vout = 0
samples = 1
cathodeone = 5
cathodetwo = 6
mapped_value = 0
fish_type= ""

# Initialize Firebase
config = {
    "apiKey": "AIzaSyD-lVlvcIy34Nr8TEgjKh01hNW9JgGmXHc",
    "authDomain": "project-fish-b839d.firebaseapp.com",
    "databaseURL": "https://project-fish-b839d-default-rtdb.firebaseio.com",
    "storageBucket": "project-fish-b839d.appspot.com"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
# Initialize Firebase Storage
storage = firebase.storage()
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(cathodeone, GPIO.OUT)
GPIO.output(cathodeone, GPIO.HIGH)
GPIO.setup(cathodetwo, GPIO.OUT)
GPIO.output(cathodetwo, GPIO.HIGH)

def set_paths_based_on_location(video_path_location):
    if video_path_location:
        # If True, set the paths accordingly
        video_path_behavior = "/home/pi/Desktop/Fish_IoT/fun2/task2/testvideo.mp4"
        video_path_count = "/home/pi/Desktop/Fish_IoT/fun3/task1/testvideo.mp4"
        death_vid = "/home/pi/Desktop/Fish_IoT/fun3/task2/testvideo.mp4" 
        image_pathpr = '/home/pi/Desktop/Fish_IoT/fun2/task3/Test_Images/captured_image.jpg'        
        image_path = '/home/pi/Desktop/Fish_IoT/fun1/tsk1/captured_image.jpg'
        Img_directory_local = "/home/pi/Desktop/Fish_IoT/fun2/task3/Test_Images/captured_image.jpg"
    else:
        # If False, set the paths differently
        video_path_behavior = "/home/pi/Desktop/Fish_IoT/Inputs/testvideo.mp4"
        video_path_count = "/home/pi/Desktop/Fish_IoT/Inputs/testvideo.mp4"
        death_vid = "/home/pi/Desktop/Fish_IoT/Inputs/testvideo.mp4"
        image_pathpr = '/home/pi/Desktop/Fish_IoT/Inputs/captured_image.jpg'
        image_path = '/home/pi/Desktop/Fish_IoT/Inputs/captured_image.jpg'
        Img_directory_local = "/home/pi/Desktop/Fish_IoT/Inputs/captured_image.jpg"

    return video_path_behavior, video_path_count, death_vid, image_pathpr, image_path,Img_directory_local

############# capture image and save ###############

def update_failure_status(failure_type, status):
    db.child('1698404487').child('hardware_failure').child(failure_type).set(status)

def capture_and_save_image(save_pathimg):
    # Open a connection to the Raspberry Pi camera
    cap = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        update_failure_status('camera_fail', True)
        return
    
    # Capture a single frame
    ret, frame = cap.read()
    
    # Check if the frame is captured successfully
    if not ret:
        print("Error: Could not capture frame.")
        update_failure_status('camera_fail', True)
        cap.release()
        return
    
    # Release the camera
    cap.release()
    
    # Save the captured frame to the specified location
    cv2.imwrite(save_pathimg, frame)
    print(f"Image saved to {save_pathimg}")
    update_failure_status('camera_fail', False)
    
save_pathimg = "/home/pi/Desktop/Fish_IoT/Inputs/captured_image.jpg"
############# End of capture image and save ##################################

#############  capture Video and save ##################################
def record_video(save_path, duration=6, fps=20, resolution=(640, 480), codec='mp4v'):
    # Initialize video capture object
    cam = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not cam.isOpened():
        print("Error: Could not open camera.")
        update_failure_status('camera_fail', True)
        return
    
    # Set the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps, resolution)
    
    # Record for the specified duration
    end_time = cv2.getTickCount() + duration * cv2.getTickFrequency()
    while cv2.getTickCount() < end_time:
        ret, frame = cam.read()
        if not ret:
            break
        # Write the frame to the video file
        out.write(frame)
    
    # Release the video capture and writer objects
    cam.release()
    out.release()
    print(f"Video recording complete. Saved to {save_path}")
    update_failure_status('camera_fail', False)

save_path = "/home/pi/Desktop/Fish_IoT/Inputs/testvideo.mp4"

#############  END of capture Video and save ##################################

##################### Start of Pregnant fish Ml part #########################

model_pathpr='/home/pi/Desktop/Fish_IoT/fun2/task3/Fish_Preg_Converted_model1.tflite'

def detect_pregnant_fish(image_pathpr,model_pathpr, threshold=0.1):
    # Load the TFLite model
    interpreter = Interpreter(model_pathpr)
    interpreter.allocate_tensors()

    input_tensor_index = interpreter.get_input_details()[0]['index']
    output = interpreter.tensor(interpreter.get_output_details()[0]['index'])

    # Load and preprocess the input image
    input_image = cv2.imread(image_pathpr)
    input_image = cv2.resize(input_image, (128, 128))
    input_image = input_image / 255.0  # Normalize to [0, 1]
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

    # Perform inference
    interpreter.set_tensor(input_tensor_index, input_image,)
    interpreter.invoke()
    yhat = output()[0][0]

    # Check the prediction
    pregnant_detected = yhat > threshold

    # Assign the result to a variable
    if pregnant_detected:
        resultpr = "Pregnant Fish Detect"
        
        Img_directory_local_up = Img_directory_local
        Img_path_local = os.path.join(Img_directory_local_up)

        try:
            storage.child('1698404487').child('Pregnant_Fish_detection').child('Pregnant Fish_Img').put(Img_path_local)
            print("Pregnant_Fish_detection uploaded to Firebase Storage")
        except Exception as e:
            print("Pregnant_Fish_detection to Firebase Storage:", e)
            
    else:
        resultpr = "Non Pregnant Fish Detect"

    return pregnant_detected, resultpr

##################### END of Pregnant fish Ml part #########################

##################### Check fish behaviral Ml part #########################

weights_path_behavior = "/home/pi/Desktop/Fish_IoT/fun2/task2/bestFFFF.pt"
fish_behavior_counts_array = []

def fish_beh(
    weights, source=None, device='cpu',save_img=False,view_img=False, save_video=True, exist_ok=False,
    line_thickness=2,
    ):

    fish_behavior_counts = {'Feeding': 0, 'Fish-Schooling': 0, 'Resting': 0, 'Swimming': 0}

    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(weights)
    model.to('cuda') if device == 'cuda' else model.to('cpu')

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

    # Output setup
    save_dir = Path('/home/pi/Desktop/Fish_IoT/fun2/task2/output') / 'exp'
    save_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = save_dir / f'{Path(source).stem}_behavior_counted.mp4'
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extract the results for the current frame
        results = model(frame)

        # Reset counts for each fish behavior in this frame
        fish_behavior_counts = {'Feeding': 0, 'Fish-Schooling': 0, 'Resting': 0, 'Swimming': 0}

        # Iterate over each result in the list
        for result in results:
            # Access the xyxy property and confidence values
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), c, class_id in zip(xyxy, conf, class_ids):
                behavior_classes = ['Feeding', 'Fish-Schooling', 'Resting', 'Swimming']

                # Convert class_id to integer
                class_id = int(class_id)
                behavior = behavior_classes[class_id]

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), line_thickness)

                # Display count on the frame at top-right corner
                fish_behavior_counts[behavior] += 1
        
        # Display the counts on the frame at top-right corner
        count_label = " | ".join([f'{behavior}: {count}' for behavior, count in fish_behavior_counts.items()])
        cv2.putText(frame, count_label, (frame_width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame with fish count
        if view_img:
            cv2.imshow('Fish Behavior Detection', frame)
            cv2.waitKey(1)
        
        # Save the frame if specified
        if save_img:
            cv2.imwrite(f'output_frame_{vid_frame_count}.jpg', frame)

        # Write the frame to the output video
        if save_video:
            video_writer.write(frame)

    video_writer.release()
    videocapture.release()
    print("Save video Done")
    cv2.destroyAllWindows()

    return fish_behavior_counts
##################### End of Check fish behaviral Ml part ##################


##################### Count the fish type in tank Ml part ##################
fish_counts = {'Goldfish': 0, 'Guppy': 0, 'Angel': 0}
weights_path_count = ("/home/pi/Desktop/Fish_IoT/fun3/task1/best.pt")

def fish_cont(weights,source=None,device='cpu',view_img=False,save_img=False,save_video=True,exist_ok=False,line_thickness=2,
        ):
    global fish_counts

    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(weights)
    model.to('cuda') if device == 'cuda' else model.to('cpu')

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

    save_dir = Path('/home/pi/Desktop/Fish_IoT/fun3/task1/output') / 'exp'
    save_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = save_dir / f'{Path(source).stem}_fish_counted.mp4'
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extract the results for the current frame
        results = model(frame)

        # Reset counts for each fish type in this frame
        fish_counts = {name: 0 for name in fish_counts}

        # Iterate over each result in the list
        for result in results:
            # Access the xyxy property and confidence values
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), c, class_id in zip(xyxy, conf, class_ids):
                fish_type = model.names[class_id]

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), line_thickness)

                # Display count on the frame at top-right corner
                fish_counts[fish_type] += 1
        
        # Display the counts on the frame at top-right corner
        count_label = " | ".join([f'{name}: {count}' for name, count in fish_counts.items()])
        cv2.putText(frame, count_label, (frame_width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame with fish count
        if view_img:
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
        
        # Save the frame if specified
        if save_img:
            cv2.imwrite(f'output_frame_{vid_frame_count}.jpg', frame)

        # Write the frame to the output video
        if save_video:
            video_writer.write(frame)

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    print("fish count video saved")
    cv2.destroyAllWindows()    
    print(fish_counts)    
################ END Count the fish type in tank Ml part##############
    
############# Ml module for fish Disease ################################
model_path1 = '/home/pi/Desktop/Fish_IoT/fun4/converted_model.tflite'
scaler_path1 = '/home/pi/Desktop/Fish_IoT/fun4/scaler_filename.joblib'
class_labels1 = {0: "Fin Rot", 1: "Red Spot", 2: "White Spot (Ich)"}

interpreter1 = tf.lite.Interpreter(model_path=model_path1)
interpreter1.allocate_tensors()

input_tensor_index1 = interpreter1.get_input_details()[0]['index']
output_tensor_index1 = interpreter1.get_output_details()[0]['index']

scaler1 = load(scaler_path1)

def predict_fish_condition(data):
    # Perform the necessary preprocessing on input data
    scaled_input = scaler1.transform(input_fun4.reshape(1, -1)).astype('float32')
    # Run inference
    interpreter1.set_tensor(input_tensor_index1,scaled_input)
    interpreter1.invoke()
    ml_output = interpreter1.get_tensor(output_tensor_index1)
    # Convert predictions to percentage without rounding
    percentage_values = np.round(ml_output.flatten() * 100, 4)

    # Format the percentage values without scientific notation
    formatted_percentage_values = [f'{val:.4f}' for val in percentage_values]

    # Prepare the response with class labels
    response = {"predictions": {label: percentage for label, percentage in zip(class_labels1.values(), formatted_percentage_values)}}

    return response

############# END Ml module for fish Disease ################################

#############  Ml module for Water qulity ######################################

model_path = '/home/pi/Desktop/Fish_IoT/fun1/tsk1/best (6).pt'

def predict_and_classify(image_path, model_path):
    # Load the YOLO model
    model = YOLO(model_path)

    # Predict on the image
    results = model(image_path)

    # Extract class names and probabilities
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()

    # Find the index with the highest probability
    max_prob_index = np.argmax(probs)
    max_prob = probs[max_prob_index]

    # Convert the result into percentage with 5 decimal points
    max_prob_percentage = format(max_prob * 100, '.5f')
     # Classify based on conditions
    if names_dict[max_prob_index] == "Clear" and float(max_prob_percentage) >= 75:
        feed = "Good"
    elif names_dict[max_prob_index] == "Dirty" and float(max_prob_percentage) >= 75:
        feed = "Bad"
    else:
        feed = "Average"

    # Print the results
    print("Predicted Class:", names_dict[max_prob_index])
    print("Predicted Class Probability (%):", max_prob_percentage)
    print("Feed Classification:", feed)
    db.child('1698404487').child('Funtion_1 Task 01').child('Predicted Class').set(names_dict[max_prob_index])
    db.child('1698404487').child('Funtion_1 Task 01').child('Predicted Class Probability (%):').set(max_prob_percentage)
    
    return feed
    
############# END of Ml module for Water qulity  ################################


############# Ml module for Clean Day count  ################################
model_path2 = '/home/pi/Desktop/Fish_IoT/fun1/converted_model.tflite'
scaler_path2 = '/home/pi/Desktop/Fish_IoT/fun1/scaler.joblib'
encoder_path2 = '/home/pi/Desktop/Fish_IoT/fun1/encoder.joblib'

interpreter2 = tf.lite.Interpreter(model_path=model_path2)
interpreter2.allocate_tensors()

input_tensor_index2 = interpreter2.get_input_details()[0]['index']
output_tensor_index2 = interpreter2.get_output_details()[0]['index']

scaler2 = load(scaler_path2)
encoder2 = load(encoder_path2)

def predict_other_conditions(data):

    # Define numerical_features and categorical_features for the ML module
    new_numerical_features = ['Species Count', 'Sunlight_Exposure', 'Feeding Frequency']
    new_categorical_features = ['Tank Size', 'Uneaten food', 'Water Quality']
    
    input_numerical = np.array([input_fun1tsk2[key] for key in new_numerical_features]).reshape(1, -1)
    input_categorical = [input_fun1tsk2[key] for key in new_categorical_features]
    input_categorical = np.array([input_fun1tsk2[key] for key in new_categorical_features]).reshape(1, -1)
    
    input_numerical_scaled = scaler2.transform(input_numerical)
    input_categorical_encoded = encoder2.transform(input_categorical).toarray()

    input_final = np.concatenate([input_numerical_scaled, input_categorical_encoded], axis=1)
    
    # Run inference
    input_final= input_final.astype('float32')
    interpreter2.set_tensor(input_tensor_index2, input_final)
    interpreter2.invoke()
    predictions = interpreter2.get_tensor(output_tensor_index2)

    # Perform rounding on predictions
    rounded_predictions = np.round(predictions.flatten()).astype(int)

    # Return the rounded predictions
    return rounded_predictions

############# End Ml module for Clean Day count  ################################


##################### Count the Death fish Ml part #############################

death_fish = {'Dead Fish': 0}
moduledeath = ("/home/pi/Desktop/Fish_IoT/fun3/task2/best (6).pt")


def death_cont(weights,source=None,device='cpu',view_img=False,save_img=False,save_video=True,exist_ok=False,line_thickness=2,
        ):
    global death_fish

    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(weights)
    model.to('cuda') if device == 'cuda' else model.to('cpu')

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

          # Output setup
    save_dir = Path('/home/pi/Desktop/Fish_IoT/fun3/task2/output') / 'exp'
    save_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = save_dir / f'{Path(source).stem}_death_counted.mp4'
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extract the results for the current frame
        results = model(frame)

        # Reset counts for each fish type in this frame
        death_fish = {name: 0 for name in death_fish}

        # Iterate over each result in the list
        for result in results:
            # Access the xyxy property and confidence values
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), c, class_id in zip(xyxy, conf, class_ids):
                fish_type = model.names[class_id]

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), line_thickness)

                # Display count on the frame at top-right corner
                death_fish[fish_type] += 1
        
        # Display the counts on the frame at top-right corner
        count_label = " | ".join([f'{name}: {count}' for name, count in death_fish.items()])
        cv2.putText(frame, count_label, (frame_width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame with fish count
        if view_img:
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
        
        # Save the frame if specified
        if save_img:
            cv2.imwrite(f'output_frame_{vid_frame_count}.jpg', frame)

        # Write the frame to the output video
        if save_video:
            video_writer.write(frame)

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    print("death fish video saved")
    cv2.destroyAllWindows()
    
    print(death_fish)

##################### End Count death fish Ml part ##################
    
############# Salinity sensor  #####################################

def capture_and_map_data(samples, cathodeone, cathodetwo, channelo, dtime):
    def map_sensor_value(Vout):
        original_min = 1
        original_max = 0
        target_min = 0.4
        target_max = 1.5
        mapped_value = ((Vout - original_min) / (original_max - original_min)) * (
                target_max - target_min) + target_min
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
        mapped_value = map_sensor_value(Vout)
        print("Salinity Value:", mapped_value, "ppt")
        db.child('1698404487').child('environmental_conditions').child('Salinity').set(mapped_value)
        
############# End Salinity sensor  ################################

###### Other Sensor funtions ####################
 
def read_temp_raw():
    with open(device_path + '/w1_slave', 'r') as f:
        valid, temp = f.readlines()
    return valid, temp


def read_temp():
    valid, temp = read_temp_raw()

    while 'YES' not in valid:
        time.sleep(0.2)
        valid, temp = read_temp_raw()

    pos = temp.index('t=')
    if pos != -1:
        # read the temperature.
        temp_string = temp[pos + 2:]
        temp_c = float(temp_string) / 1000.0
        temp_f = temp_c * (9.0 / 5.0) + 32.0
        return temp_c, temp_f
        print(' ROM: ' + rom)

def read_bh1750():
    with smbus2.SMBus(3) as bus:
        bus.write_byte(bh1750_address, 0x10)
        time.sleep(0.5)
        data = bus.read_i2c_block_data(bh1750_address, 0x00, 2)
        lux = (data[1] + (256 * data[0])) / 1.2
    return lux

###### End of Other Sensor funtions ####################

def check_wifi_status():
    result = subprocess.run(['iwgetid', '-r'], capture_output=True, text=True)
    if result.returncode == 0:
        return True
    else:
        return False
          
############ Strt Main loop #################
    
while True:
    wifi_status = check_wifi_status()
    # Check WiFi status
    if not wifi_status:  
        # WiFi is not connected, indicate bulb on GPIO 26
        GPIO.setup(26, GPIO.OUT)
        GPIO.output(26, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(26, GPIO.LOW)
        print("WiFi Connection Lost")
        time.sleep(0.5)
        continue
 
    # data download function
    if time.monotonic() - time_for_download_data > 10 or time_for_download_data == 0:
        time_for_download_data = time.monotonic()
        Feeding_Frequency = float(db.child("1698404487").child('Funtion_1 Task 02').child("input").child('Feeding Frequency').get().val())
        Species_Count = float(db.child("1698404487").child('Funtion_1 Task 02').child("input").child('Species Count').get().val())
        Tank_Size = str(db.child("1698404487").child('Funtion_1 Task 02').child("input").child('Tank Size').get().val())
        Uneaten_food = str(db.child("1698404487").child('Funtion_1 Task 02').child("input").child('Uneaten Food').get().val())
    
        print("Feeding_Frequency:", Feeding_Frequency)
        print("Species_Count:", Species_Count)
        print("Tank_Size:", Tank_Size)
        print("Uneaten_food:", Uneaten_food)
    
        print("Data download done")
        GPIO.setup(26, GPIO.OUT)
        GPIO.output(26, GPIO.HIGH)
        time.sleep(0.2)
        GPIO.output(26, GPIO.LOW)
        time.sleep(0.1)
    else:
        print("Error retrieving Feeding_Frequency data")
    
   
    # Get the value of video_path_location from your database
    video_path_location = db.child("1698404487").child('Video_Path').get().val()

    # Call the function to set the paths
    video_path_behavior, video_path_count, death_vid, image_pathpr, image_path,Img_directory_local = set_paths_based_on_location(video_path_location)

 
    # Capture the video
    record_video(save_path)
    GPIO.setup(26, GPIO.OUT)
    GPIO.output(26, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(26, GPIO.LOW)
    time.sleep(30)
    
    
    #Salinity sensor
    try:
        capture_and_map_data(samples, cathodeone, cathodetwo, channelo, dtime)
        SLNTY_failure = False
        GPIO.setup(26, GPIO.OUT)
        GPIO.output(26, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(26, GPIO.LOW)
        time.sleep(5)
    except Exception as e:
        SLNTY_failure = True
        print("Salinity Fail:", e)
        continue

    # Check Temperature sensor status
    try:
        c, f = read_temp()
        TMP_failure = False
        print('C={:,.3f} F={:,.3f}'.format(c, f))
        GPIO.setup(26, GPIO.OUT)
        GPIO.output(26, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(26, GPIO.LOW)
        time.sleep(5)
    except Exception as e:
        TMP_failure = True
        print("Temperature Sensor Fail:", e)
        continue

    # Check ADC status
    try:
        voltage = channel.voltage
        ADC_failure = False
        ph = 7 - (voltage - 2.5) / 0.5
        print("pH Value: ", ph)
        GPIO.setup(26, GPIO.OUT)
        GPIO.output(26, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(26, GPIO.LOW)
        time.sleep(5)
    except Exception as e:
        ADC_failure = True
        print("ADC Fail:", e)
        continue
     # Capture the image
     
    capture_and_save_image(save_pathimg)
    
    GPIO.setup(26, GPIO.OUT)
    GPIO.output(26, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(26, GPIO.LOW)
    time.sleep(5)
    
    # Calculate the dissolved oxygen concentration using the Weiss equation
    DO = 6.046 - 0.1566 * c + 0.002981 * c ** 2 - 0.00000598 * c ** 3 + (0.4154 - 0.003022 * c) * (
            mapped_value) ** 0.5
    print("Dissolved oxygen concentration:", DO, "mg/L")
    GPIO.setup(26, GPIO.OUT)
    GPIO.output(26, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(26, GPIO.LOW)
    time.sleep(5)
    
    # bh1750 sensor reading function
    try:
        light_intensity = read_bh1750()
        light_sensor_failure = False
        print("Light Intensity:{:.2f}".format(light_intensity))
        GPIO.setup(26, GPIO.OUT)
        GPIO.output(26, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(26, GPIO.LOW)
    except Exception as e:
        light_sensor_failure = True
        print("light fail", light_sensor_failure)
        
    time.sleep(5)
    
    # Feed sensor values to the existing Funtion4 ML Module Fish conditions
    input_fun4 = np.array([c, 90, DO, ph, 600, 0.03, mapped_value])
    ml_predictions =predict_fish_condition(input_fun4)
    print("Funtion 4 output:", ml_predictions)
    GPIO.setup(26, GPIO.OUT)
    GPIO.output(26, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(26, GPIO.LOW)
    
    time.sleep(5)
    
    
    # water qulity ML Module call
    feed=predict_and_classify(image_path, model_path)
    GPIO.setup(26, GPIO.OUT)
    GPIO.output(26, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(26, GPIO.LOW)
    time.sleep(5)
    
    
    # feed the data to Clean tank funtion   
    input_fun1tsk2 = {
        'Species Count': [Species_Count],
        'Sunlight_Exposure': [light_intensity],
        'Feeding Frequency': [Feeding_Frequency],
        'Tank Size': [Tank_Size],
        'Uneaten food': [Uneaten_food],
        'Water Quality': [feed]
    }

    ml_predictionsfun1_ts2 = predict_other_conditions(input_fun1tsk2)
    ml_predictionsfun1_ts2_list = ml_predictionsfun1_ts2.tolist()
    print("clean tank withing:", ml_predictionsfun1_ts2_list)
    GPIO.setup(26, GPIO.OUT)
    GPIO.output(26, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(26, GPIO.LOW)
    
    time.sleep(5)
    
    
    ######## Call fish_count function #####################
    
    fish_cont(weights=weights_path_count, source=video_path_count, device='cpu',view_img=True, save_img=False, save_video=True)
    time.sleep(5)
    #print(fish_counts)
    GPIO.setup(26, GPIO.OUT)
    GPIO.output(26, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(26, GPIO.LOW)
    time.sleep(5)
    
    video_directory_local = "/home/pi/Desktop/Fish_IoT/fun3/task1/output/exp"
    video_name = "testvideo_fish_counted.mp4"
    video_path_local = os.path.join(video_directory_local, video_name)

    try:
        storage.child('1698404487').child('fish_detection').child('fish_detection_video').put(video_path_local)
        print("fish_detection video uploaded to Firebase Storage")
    except Exception as e:
        print("Error uploading video to Firebase Storage:", e)
        
    
    #Call pregnant Fish detect ML module 
    pregnant_detected, resultpr = detect_pregnant_fish(image_pathpr,model_pathpr)
    print(resultpr)
    GPIO.setup(26, GPIO.OUT)
    GPIO.output(26, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(26, GPIO.LOW)
    time.sleep(5)
    
    #Call the Dead fish count funtion
    death_cont(weights=moduledeath, source=death_vid, device='cpu',view_img=True, save_img=False, save_video=True)
    time.sleep(5)
    GPIO.setup(26, GPIO.OUT)
    GPIO.output(26, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(26, GPIO.LOW)
    time.sleep(5)
    
    video_directory_local = "/home/pi/Desktop/Fish_IoT/fun3/task2/output/exp"
    video_name = "testvideo_death_counted.mp4"
    video_path_local = os.path.join(video_directory_local, video_name)

    try:
        storage.child('1698404487').child('Dead_fish_detection').child('Dead_fish_video').put(video_path_local)
        print("Dead fish video uploaded to Firebase Storage")
    except Exception as e:
        print("Error uploading video to Firebase Storage:", e)
    
    
    
    #Call Ml module for check fish behaviral 
    result_counts = fish_beh(weights=weights_path_behavior, view_img=True,source=video_path_behavior, device='cpu', save_img=False, save_video=True)
    time.sleep(5)
    print(result_counts)
    GPIO.setup(26, GPIO.OUT)
    GPIO.output(26, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(26, GPIO.LOW)
    time.sleep(5)
    
    video_directory_local = "/home/pi/Desktop/Fish_IoT/fun2/task2/output/exp"
    video_name = "testvideo_behavior_counted.mp4"
    video_path_local = os.path.join(video_directory_local, video_name)

    try:
        storage.child('1698404487').child('fish_behaviral').child('fish_behaviral_video').put(video_path_local)
        print("Fish behavior video uploaded to Firebase Storage")
    except Exception as e:
        print("Error uploading video to Firebase Storage:", e)
    
        # Append the result to fish_behavior_counts_array
    fish_behavior_counts_array.append(result_counts)
    print(fish_behavior_counts_array)
    max_results = 5
    if len(fish_behavior_counts_array) >= max_results:
        print("max_result")
        last_five_results = fish_behavior_counts_array[-max_results:]
        print(last_five_results)
        # Access the dictionary inside the list
        result_dict = last_five_results[0]

        # Extract individual values
        feeding_count = result_dict['Feeding']
        schooling_count = result_dict['Fish-Schooling']
        resting_count = result_dict['Resting']
        swimming_count = result_dict['Swimming']
        
        # Now, you have individual variables containing the counts
        print("Feeding Count:", feeding_count)
        print("Fish-Schooling Count:", schooling_count)
        print("Resting Count:", resting_count)
        print("Swimming Count:", swimming_count)

        # Calculate total count
        total_count = feeding_count + schooling_count + resting_count + swimming_count
        # Print or use the total count as needed
        print("Total Count:", total_count)
        
        # Check if resting_count/feeding_count is equal to a specific value
        ratio_condition = resting_count / feeding_count if feeding_count != 0 else 0

        if ratio_condition == 0:
            behavioral = False
            print("Resting Count / Feeding Count ratio is equal to 0:")
            print(behavioral)
        else:
            behavioral = True
            print(f"Resting Count / Feeding Count ratio is NOT equal to 0. Behavioral variable set to True.:")
            print(behavioral)
            # Your additional logic for the else case goes here
        
         # Check if Fish-Schooling count is equal in all elements and not equal to 0
        scl_behavr = all(result['Fish-Schooling'] == last_five_results[0]['Fish-Schooling'] != 0 for result in last_five_results)

        if scl_behavr:
            print("Fish-Schooling count is equal in all elements and not equal to 0. scl_behavr variable set to True.")
            print(scl_behavr)
            
            # Your additional logic for the True case goes here
        else:
            print("Fish-Schooling count is NOT equal in all elements or equal to 0. scl_behavr variable set to False.")
            print(scl_behavr)
            # Your additional logic for the False case goes here
        # Reset max_results
        max_results = 0
        # Reset fish_behavior_counts_array to an empty list
        fish_behavior_counts_array = []
        db.child('1698404487').child('Fish Unusual').child('Long_time_rest').set(behavioral)
        db.child('1698404487').child('Fish Unusual').child('Fish_Schooling').set(scl_behavr)
        GPIO.setup(26, GPIO.OUT)
        GPIO.output(26, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(26, GPIO.LOW)
        time.sleep(5)

    else:
        print("Condition not met. Continue loop.")
    
    
    ######## data upload function #############
    if time.monotonic() - time_for_upload_data > 10:
        time_for_upload_data = time.monotonic()
        db.child('1698404487').child('environmental_conditions').child('pH').set(ph)
        db.child('1698404487').child('environmental_conditions').child('Dissolved oxygen').set(DO)
        db.child('1698404487').child('environmental_conditions').child('light_intensity').set(light_intensity)
        db.child('1698404487').child('environmental_conditions').child('temperature').set(c)
        db.child('1698404487').child('Funtion 4').set(ml_predictions)
        db.child('1698404487').child('Funtion_1 Task 02').child('Clean tank days').set(ml_predictionsfun1_ts2_list)
        db.child('1698404487').child('Function 2 Task 03').child('pregnant Fish').set(resultpr)
        db.child('1698404487').child('fish_detection').set(fish_counts)
        db.child('1698404487').child('Death_fish_detection').set(death_fish)
        db.child('1698404487').child('Fish Behavior').set(result_counts)
        db.child('1698404487').child('hardware_failure').child('light_sensor_failure').set(light_sensor_failure)
        db.child('1698404487').child('hardware_failure').child('ADC_Failure').set(ADC_failure)
        db.child('1698404487').child('hardware_failure').child('Temperature_Failure').set(TMP_failure)
        db.child('1698404487').child('hardware_failure').child('Salinity_Failure').set(SLNTY_failure)
            
        print("data upload done")
        print("---------------------------------")
        GPIO.setup(26, GPIO.OUT)
        GPIO.output(26, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(26, GPIO.LOW)
        GPIO.setup(26, GPIO.OUT)
        GPIO.output(26, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(26, GPIO.LOW)
        GPIO.setup(26, GPIO.OUT)
        GPIO.output(26, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(26, GPIO.LOW)
        time.sleep(15)

    else:
         time.sleep(5)

