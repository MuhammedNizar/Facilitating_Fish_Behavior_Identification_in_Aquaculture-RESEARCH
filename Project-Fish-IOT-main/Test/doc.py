from ultralytics import YOLO
import cv2
import numpy as np

image = cv2.imread("/home/pi/Desktop/Fish_IoT/fun1/new.jpg")
model = YOLO('/home/pi/Desktop/Fish_IoT/fun1/tsk1/last (2).pt')

original_shape = image.shape
print(f"Original image shape: {original_shape}")

resized_image = cv2.resize(image, dsize=(64, 64))
print(f"Resized image shape: {resized_image.shape}")

if len(original_shape) == 2:  # Grayscale image
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)


final_image = np.expand_dims(resized_image, axis=0)
print(f"Final image shape: {final_image.shape}")

results = model(final_image)

# Run inference on the source
#results = model(source, stream=True)  # generator of Results objects

names_dict = model.names

probs = results.xyxy[0][:, -1].cpu().numpy().tolist()

print("Class Names:", names_dict)
print("Probabilities:", probs)

# Find the index with the highest probability
max_prob_index = np.argmax(probs)
max_prob = probs[max_prob_index]

# Convert the result into percentage with 5 decimal points
max_prob_percentage = format(max_prob * 100, '.5f')

print("Predicted Class:", names_dict[max_prob_index])
print("Predicted Class Probability (%):", max_prob_percentage)
