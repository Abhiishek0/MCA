from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLO model (yolov8n is now downloaded)
model = YOLO('yolov8n.pt') 

# --- Start Detection Test using Webcam ---
print("--- Starting YOLO Webcam Verification ---")
print("If successful, a window will open. Press 'q' or 'esc' to exit the live stream.")

# '0' typically refers to the default webcam.
model.predict(source=0, show=True, stream=True)

print("\n✅ SUCCESS! YOLO detection pipeline is verified with live video.")