# code/yolo_detection.py

from ultralytics import YOLO
import cv2
import os

# -----------------------------------------------------------
# 1. SETUP & MODEL LOADING
# -----------------------------------------------------------

# YOLO Model Load karna hai (Jo aapke MCA folder mein downloaded hai)
YOLO_MODEL = YOLO('../yolov8n.pt') 

# Output path jahan cropped images save honge for CNN training
OUTPUT_DIR = '../data/processed/temp_crops/' 

def setup_directories():
    """Output directories banana."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

def process_video_and_crop(video_source=0):
    """Live video se frames lena aur humans ko crop karna."""
    setup_directories()
    
    # Live webcam stream start karna
    cap = cv2.VideoCapture(video_source)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO se detection run karna
        results = YOLO_MODEL.predict(source=frame, verbose=False) 

        # Sirf 'person' (Class ID 0) ko check karna
        for r in results:
            for box in r.boxes:
                if box.cls.item() == 0: # Check if the detected object is a person
                    # Bounding box coordinates nikalna
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Human ko frame se crop karna
                    cropped_person = frame[y1:y2, x1:x2]
                    
                    # Cropped image ko save karna (Yahi data CNN ko milega)
                    # Note: Yahan par aapko sequence banani hogi, jo hum baad mein dekhenge.
                    save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count}.jpg")
                    cv2.imwrite(save_path, cropped_person)
                    frame_count += 1
                    
        # Stop condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing stopped.")

# Uncomment this line jab aap test karna chahein:
# process_video_and_crop(video_source=0)