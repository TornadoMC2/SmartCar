import cv2
import numpy as np
import urllib.request
import os  # Added for checking file path

# --- Configuration ---
esp32_ip = "192.168.4.1"
# --- !! UPDATE PORT HERE !! ---
esp32_port = 81
stream_endpoint = "/stream"

# Construct the full stream URL
stream_url = f"http://{esp32_ip}:{esp32_port}{stream_endpoint}"

# --- Face Detection Setup ---
# Path to the Haar Cascade file.
# Place the XML file in the same folder as the script, or provide the full path.
cascade_file_name = 'haarcascade_frontalface_default.xml'

# Check if the cascade file exists
if not os.path.exists(cascade_file_name):
    print(f"Error: Cascade file not found at '{cascade_file_name}'")
    print("Please download 'haarcascade_frontalface_default.xml' and place it")
    print("in the same directory as this script, or update the path.")
    exit()

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cascade_file_name)
if face_cascade.empty():
    print(f"Error: Could not load cascade classifier from '{cascade_file_name}'")
    exit()

print(f"Attempting to connect to stream: {stream_url}")

# --- Video Stream Handling ---
try:
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"Error: Cannot open stream at {stream_url}")
        # Add specific checks as before...
        exit()

    print("Stream opened successfully. Detecting faces... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()  # Read one frame

        if not ret or frame is None:  # Added check for None frame
            print("Error: Failed to receive frame or frame is empty. Stream might have ended.")
            # Optional: Try to reconnect or simply break
            break

            # --- Face Detection Logic ---
        # 1. Convert the frame to grayscale (Haar cascades work better on grayscale)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Detect faces in the grayscale frame
        #    - scaleFactor: How much the image size is reduced at each image scale. 1.1 is common.
        #    - minNeighbors: How many neighbors each candidate rectangle should have to retain it. Higher value = fewer detections but higher quality. 5 is common.
        #    - minSize: Minimum possible object size. Objects smaller than this are ignored.
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)  # You can adjust minSize depending on expected face size
        )

        # 3. Draw rectangles around the detected faces on the original color frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

        # --- Display ---
        # Display the resulting frame (with rectangles if faces are detected)
        cv2.imshow('ESP32 Camera Stream - Face Detection', frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # When everything done, release the capture and destroy windows
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")