import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

alphabet_classes = [chr(i) for i in range(65, 91)]  # List of characters from A to Z
dataset_size = 100

# Attempt to find the correct camera index
index = 0
cap = None
while True:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera found at index {index}")
        break
    cap.release()
    index += 1
    if index > 10:  # Limit to 10 attempts
        print("No available camera found")
        exit()

for letter in alphabet_classes:
    class_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {letter}')

    # Wait for the user to be ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        cv2.putText(frame, f'Get ready for {letter}! Press "Q" to start.', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect dataset_size images for the current letter
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 0

cap.release()
cv2.destroyAllWindows()
