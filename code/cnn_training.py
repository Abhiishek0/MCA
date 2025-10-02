
# --- 1. CONFIGURATION AND FILE PATHS (Update this section) ---

# Data Paths (Should point to the 'train' folder containing 'walking' and 'fighting')
DATA_PATH = '../data/processed/train/' 

# Model Parameters
SEQUENCE_LENGTH = 30  # Number of frames (images) per sequence. (e.g., 30 frames = 1 second of video)
IMAGE_WIDTH, IMAGE_HEIGHT = 64, 64 # Size to resize the cropped images to.
NUM_CLASSES = 2 # 0: Normal (Walking), 1: Abnormal (Fighting)

# code/cnn_training.py (Add this function)

import cv2
import glob # For finding files using patterns
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ... (rest of the imports and config)

# --- NEW FUNCTION: DATA PREPARATION ---

def create_sequences_and_labels(data_path, sequence_length, img_size):
    """
    Loads cropped images, groups them into time-ordered sequences, 
    and generates corresponding one-hot encoded labels.
    """
    X = []  # To hold the image sequences (our features)
    y = []  # To hold the labels (0 or 1)
    
    # Define your activity labels here (adjust based on your collected data)
    CLASSES = sorted(os.listdir(data_path))
    CLASSES = [c for c in CLASSES if os.path.isdir(os.path.join(data_path, c))] # Only keep folders
    
    # Create a mapping from label name to integer (e.g., 'walking': 0, 'fighting': 1)
    label_map = {label: i for i, label in enumerate(CLASSES)}
    print(f"Detected Classes and Mapping: {label_map}")

    # Loop through each activity folder (e.g., 'walking', 'fighting')
    for label in CLASSES:
        label_dir = os.path.join(data_path, label)
        
        # Get all image file paths (sorted by frame number is CRITICAL for time order)
        all_image_paths = sorted(glob.glob(os.path.join(label_dir, f"{label}_*.jpg")))
        
        # We need at least 'sequence_length' images to make one sequence
        if len(all_image_paths) < sequence_length:
            print(f"Skipping {label}: Not enough images ({len(all_image_paths)} < {sequence_length})")
            continue
            
        print(f"Processing {label} with {len(all_image_paths)} images...")
        
        # Create sequences using a sliding window approach
        for i in range(0, len(all_image_paths) - sequence_length + 1, sequence_length // 2): 
            # Sequence: A list of 30 image paths
            sequence_paths = all_image_paths[i : i + sequence_length]
            
            # Load images for the current sequence
            sequence_data = []
            for img_path in sequence_paths:
                img = cv2.imread(img_path)
                
                # Check if image load failed or is corrupted
                if img is None:
                    continue 

                # Resize to the standard size (64x64) and normalize
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize pixel values to 0-1
                sequence_data.append(img)

            # Only add the sequence if it has the full length
            if len(sequence_data) == sequence_length:
                X.append(np.array(sequence_data))
                y.append(label_map[label])

    # Convert lists to NumPy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    # Convert integer labels to one-hot encoding (required by categorical_crossentropy loss)
    y = to_categorical(y, num_classes=len(CLASSES))

    return X, y, len(CLASSES), label_map


# -----------------------------------------------------------
# 5. EXECUTION BLOCK (Update this section)
# -----------------------------------------------------------

if __name__ == "__main__":
    
    # --- 1. Load and Prepare Data ---
    # NOTE: Run this *after* your team has collected both 'walking' and 'fighting' data.
    
    # Example: If you have a 'walking' folder, this will run.
    try:
        X, y, num_classes, label_map = create_sequences_and_labels(
            DATA_PATH, SEQUENCE_LENGTH, (IMAGE_HEIGHT, IMAGE_WIDTH)
        )

        print("\n--- Data Preparation Summary ---")
        print(f"Total Sequences Created (X): {X.shape}") 
        # Output should look like: (Total_Sequences, 30, 64, 64, 3)
        print(f"Total Labels Created (y): {y.shape}")
        
        # --- 2. Split Data for Training and Testing ---
        # This prevents overfitting and evaluates model performance honestly.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Training Data Samples: {X_train.shape[0]}")
        print(f"Testing Data Samples: {X_test.shape[0]}")
        
        # --- 3. Model Creation and Training ---
        # model = create_cnn_lstm_model(X.shape[1:], num_classes) # Use X_train.shape[1:] for input shape
        # print(model.summary())
        
        # history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
        # model.save('../models/abnormal_detection_cnn_lstm.h5')
        
        print("\nReady for model training (Model Fit code is commented out).")
        
    except FileNotFoundError:
        print(f"Error: Data folder not found at {DATA_PATH}. Please collect data first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")