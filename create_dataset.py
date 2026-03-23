import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
EXPECTED_FEATURE_SIZE = 42

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return None, None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        print(f"No hand detected in image: {img_path}")
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    x_ = [landmark.x for landmark in hand_landmarks.landmark]
    y_ = [landmark.y for landmark in hand_landmarks.landmark]
    min_x = min(x_)
    min_y = min(y_)

    data_aux = []
    for landmark in hand_landmarks.landmark:
        data_aux.append(landmark.x - min_x)
        data_aux.append(landmark.y - min_y)

    if len(data_aux) != EXPECTED_FEATURE_SIZE:
        print(f"Skipping {img_path}: unexpected feature size {len(data_aux)}")
        return None, None

    return data_aux, os.path.basename(os.path.dirname(img_path))

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        print("Please run the collect_imgs.py script first to collect images.")
        return

    for dir_ in sorted(os.listdir(DATA_DIR)):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue
        
        print(f"Processing images for class: {dir_}")
        for img_name in sorted(os.listdir(dir_path)):
            _, ext = os.path.splitext(img_name)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            img_path = os.path.join(dir_path, img_name)
            data_aux, label = process_image(img_path)
            
            if data_aux is not None and label is not None:
                data.append(data_aux)
                labels.append(label)

    if not data or not labels:
        print("No valid data was processed. Please check your images.")
        return

    data_array = np.array(data, dtype=np.float32)
    labels_array = np.array(labels)

    print(f"Data shape: {data_array.shape}")
    print(f"Labels shape: {labels_array.shape}")

    # Save the processed data and labels into a pickle file
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data_array, 'labels': labels_array, 'feature_size': EXPECTED_FEATURE_SIZE}, f)

    print(f"Dataset created with {len(data)} samples.")
    print("data.pickle file has been created successfully.")

if __name__ == "__main__":
    main()
