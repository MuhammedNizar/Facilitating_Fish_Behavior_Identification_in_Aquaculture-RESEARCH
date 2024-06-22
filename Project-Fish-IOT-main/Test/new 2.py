from ultralytics import YOLO
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms


#names = {0:'Clear', 1:'Dirty'}

try:
    
    img = Image.open('/home/pi/Desktop/Fish_IoT/fun1/image.jpg')
    # Assuming `im` is your Image object
    im_array = np.array(img)

# Now you can access the shape attribute
    print(im_array.shape)
    
    print("123")
    #img.show()
    #Resize the image
    img_resized = img.resize((64, 64))
   

    model = YOLO('/home/pi/Desktop/Fish_IoT/fun1/tsk1/last (2).pt')
    
    print("1")
    
    width, height = img_resized.size
    print(f"Width: {width}, Height: {height}")

    #model(img_resized)
    
    numpy_image = np.array(img_resized)
    print(numpy_image.shape)
    
    
      # Ensure the image has 3 channels (for RGB)
    if len(numpy_image.shape) == 2:
        numpy_image = np.stack((numpy_image,)*3, axis=-1)
        
    # Add an extra dimension at the beginning
    final_imag = np.expand_dims(numpy_image, axis=0)
    
    final_image = np.transpose(final_imag, (0, 3, 1, 2))

    print(final_image.shape)  # Should print: (1, 3, 64, 64)    
    
    results = model(final_image,stream=True,)
    #results = model(img, stream=True,)
    #results = model('/home/pi/Desktop/Fish_IoT/fun1/image.jpg') 
    #model('/home/pi/Desktop/Fish_IoT/fun1/new.jpg')
    print("2")
       
    names_dict = model.names
    print("Class Names:", names_dict)
    print("3")
    
    for result in results:
        probs = results.xyxy[0][:, -1].tolist()
        print("Probabilities:", probs)
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



 

