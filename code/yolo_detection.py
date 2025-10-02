# code/yolo_detection.py
# This script handles real-time human detection and cropping for CNN feature extraction.

from ultralytics import YOLO
import cv2
import os
import time

# --- 1. CONFIGURATION ---

# Path to your YOLO model (adjust if yolov8n.pt is in a different place)
# NOTE: We use '../' because this script is inside the 'code' directory.
YOLO_MODEL_PATH = '../yolov8n.pt' 

# Output directory for saving the cropped images. 
# We'll save 'Normal' and 'Abnormal' data separately for CNN training.
OUTPUT_BASE_DIR = '../data/processed/train/'

# Define the class ID for 'person' in the COCO dataset (usually 0)
PERSON_CLASS_ID = 0 

# --- 2. SETUP FUNCTIONS ---

def setup_directories(activity_label):
    """Creates the necessary output folders (e.g., ../data/processed/train/fighting/ or ../data/processed/train/walking/)."""
    
    # Example: '../data/processed/train/walking/'
    output_dir = os.path.join(OUTPUT_BASE_DIR, activity_label)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

# --- 3. MAIN PROCESSING FUNCTION ---

def process_video_and_crop(video_source, activity_label):
    """
    Reads a video source, detects humans, crops the ROI, and saves the image.

    Args:
        video_source (str/int): Path to video file or 0 for webcam.
        activity_label (str): Folder name (e.g., 'normal', 'fighting').
    """
    
    output_dir = setup_directories(activity_label)
    
    # Load the YOLO model
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    frame_count = 0
    print(f"\n--- Starting Detection and Cropping for '{activity_label}' ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or failed.")
            break

        # 1. Run YOLO Detection (verbose=False keeps the output clean)
        results = yolo_model.predict(source=frame, verbose=False) 

        # 2. Process Detection Results
        for r in results:
            for box in r.boxes:
                # Check if the detected object is a person
                if box.cls.item() == PERSON_CLASS_ID:
                    
                    # Get bounding box coordinates (normalized to frame size)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Ensure bounding box coordinates are valid
                    # Add a small buffer around the person (e.g., 10 pixels)
                    x1 = max(0, x1 - 10)
                    y1 = max(0, y1 - 10)
                    x2 = min(frame.shape[1], x2 + 10)
                    y2 = min(frame.shape[0], y2 + 10)

                    # 3. Crop the Human Image (Region of Interest)
                    cropped_person = frame[y1:y2, x1:x2]
                    
                    # If the crop is valid (not empty), save it
                    if cropped_person.size > 0:
                        
                        # Save path: e.g., ../data/processed/train/fighting/0001.jpg
                        save_path = os.path.join(output_dir, f"{activity_label}_{frame_count:05d}.jpg")
                        cv2.imwrite(save_path, cropped_person)
                        frame_count += 1
                        
                        # Optional: Draw the bounding box on the original frame for display
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add counter on screen
                        cv2.putText(frame, f"Frames: {frame_count}", (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        
        # Display the result frame
        cv2.imshow('CCTV Detector - Press Q to Stop', frame)
        
        # Stop condition: Press 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCompleted! Saved {frame_count} cropped images to {output_dir}")

# --- 4. EXECUTION BLOCK ---

if __name__ == "__main__":
    
    # --- TASK 1: Collect Normal Data ---
    # NOTE: Run this first! You will stand/walk in front of your camera.
    # process_video_and_crop(video_source=0, activity_label='normal')
    
    # --- TASK 2: Collect Abnormal Data (e.g., Fighting) ---
    # You will act out a fighting motion or use a video clip (path to video file)
    # process_video_and_crop(video_source=0, activity_label='fighting')
    
    
    # EXAMPLE: Uncomment the line below to test live capture for 'normal' activity:
    process_video_and_crop(video_source=0, activity_label='walking')