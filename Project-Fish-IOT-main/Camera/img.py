import cv2

def capture_and_save_image(save_path):
    # Open a connection to the Raspberry Pi camera
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture a single frame
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Could not capture frame.")
        cap.release()
        return

    # Release the camera
    cap.release()

    # Save the captured frame to the specified location
    cv2.imwrite(save_path, frame)

    print(f"Image saved to {save_path}")

# Example usage
save_path = "/home/pi/Desktop/Fish_IoT/Inputs/captured_image.jpg"
capture_and_save_image(save_path)
