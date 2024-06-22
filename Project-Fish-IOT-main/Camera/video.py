import cv2

def record_video(save_path, duration=6, fps=20, resolution=(640, 480), codec='mp4v'):
    # Initialize video capture object
    cam = cv2.VideoCapture(0)

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

# Example usage
save_path = "/home/pi/Desktop/Fish_IoT/Inputs/testvideo.mp4"
record_video(save_path)
