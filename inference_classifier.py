import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import argparse

# Disable TensorFlow OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def find_available_camera(preferred_index: int = 0, max_index: int = 10):
    if preferred_index is not None:
        cap = cv2.VideoCapture(preferred_index)
        if cap.isOpened():
            return cap, preferred_index
        cap.release()

    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap, index
        cap.release()

    return None, None


def extract_hand_features(hand_landmarks, expected_feature_size):
    x_ = [landmark.x for landmark in hand_landmarks.landmark]
    y_ = [landmark.y for landmark in hand_landmarks.landmark]
    min_x = min(x_)
    min_y = min(y_)

    data_aux = []
    for landmark in hand_landmarks.landmark:
        data_aux.append(landmark.x - min_x)
        data_aux.append(landmark.y - min_y)

    if len(data_aux) < expected_feature_size:
        data_aux.extend([0.0] * (expected_feature_size - len(data_aux)))
    elif len(data_aux) > expected_feature_size:
        data_aux = data_aux[:expected_feature_size]

    return np.asarray(data_aux, dtype=np.float32), x_, y_


def main(args):
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        print("Please run the train_classifier.py script first to generate the model.")
        return

    try:
        with open(args.model_path, 'rb') as f:
            model_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model_dict['model'] if isinstance(model_dict, dict) and 'model' in model_dict else model_dict
    trained_classes = model_dict.get('classes') if isinstance(model_dict, dict) else None
    expected_feature_size = int(model_dict.get('feature_size', 42)) if isinstance(model_dict, dict) else 42

    if trained_classes is None and hasattr(model, 'classes_'):
        trained_classes = [str(c) for c in model.classes_.tolist()]

    cap, camera_index = find_available_camera(args.camera_index)
    if cap is None:
        print("Error: No available camera found.")
        return

    print(f"Using camera index: {camera_index}")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=args.min_detection_confidence)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            features, x_, y_ = extract_hand_features(hand_landmarks, expected_feature_size)
            prediction_output = model.predict([features])[0]
            predicted_character = str(prediction_output)

            if trained_classes and predicted_character not in trained_classes:
                predicted_character = "Unknown"

            x1 = max(0, int(min(x_) * w) - 10)
            y1 = max(0, int(min(y_) * h) - 10)
            x2 = min(w, int(max(x_) * w) + 10)
            y2 = min(h, int(max(y_) * h) + 10)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.putText(frame, predicted_character, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run real-time sign language inference using webcam.')
    parser.add_argument('--model_path', type=str, default='./model.p', help='Path to the trained model pickle file.')
    parser.add_argument('--camera_index', type=int, default=0, help='Preferred camera index to open.')
    parser.add_argument('--min_detection_confidence', type=float, default=0.3, help='MediaPipe min detection confidence (0-1).')

    args = parser.parse_args()
    main(args)
