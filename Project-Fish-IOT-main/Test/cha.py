from ultralytics import YOLO
import numpy as np
from PIL import Image

names = {0: 'Clear', 1: 'Dirty'}

try:
    
    img = Image.open('/home/pi/Desktop/Fish_IoT/fun1/image.jpg')
    #img.show()
    #Resize the image
    img_resized = img.resize((640, 640))
    print("1")
    
    width, height = img_resized.size
    print(f"Width: {width}, Height: {height}")

    model = YOLO('/home/pi/Desktop/Fish_IoT/fun1/last.pt')

    print("1")
    img_array = np.array(img_resized)
    img_list = [img_array]
    results = model(img_array, stream=True)

    print("2")
    
    for result in results:
        names_dict = result.names
        print("3")
        probs = result.xyxy[0][:, -1].tolist()
        print("4")

        # Find class with highest probability
        max_prob_index = np.argmax(probs)
        print("5")
        max_prob = probs[max_prob_index]
        print("6")
        max_prob_percentage = format(max_prob * 100, '.5f')
        print("7")
        print("Class Names:", names_dict)
        print("Probabilities:", probs)
        print("Predicted Class:", int(names_dict[max_prob_index]))
        print("Predicted Class Probability (%):", max_prob_percentage)

except Exception as e:
    print(f"An error occurred: {e}")
