import cv2
import numpy as np
import os
import pickle

# Try different device indices or use the URL provided by DroidCam
# video = cv2.VideoCapture(0)  # Default laptop camera
# video = cv2.VideoCapture(1)  # Try device index 1
# video = cv2.VideoCapture(2)  # Try device index 2
video = cv2.VideoCapture(1)  # Use the URL provided by DroidCam

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_data = []
i = 0

name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture video. Check DroidCam connection.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))

        if len(faces_data) < 50 and i % 10 == 0:
            faces_data.append(resized_img)

        i += 1
        cv2.putText(frame, f"Captured: {len(faces_data)}/50", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or len(faces_data) == 50:
        print("Exiting program.")
        break

video.release()
cv2.destroyAllWindows()

# Convert and reshape face data
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), -1)

# Ensure "data" directory exists
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Save names
names_file = os.path.join(data_dir, 'names.pkl')
faces_file = os.path.join(data_dir, 'faces_data.pkl')

# Load existing data if present
if os.path.exists(names_file):
    try:
        with open(names_file, 'rb') as f:
            names = pickle.load(f)
    except EOFError:
        names = []
else:
    names = []

names += [name] * len(faces_data)
with open(names_file, 'wb') as f:
    pickle.dump(names, f)

# Save face data
if os.path.exists(faces_file):
    try:
        with open(faces_file, 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
    except EOFError:
        faces = faces_data
else:
    faces = faces_data

with open(faces_file, 'wb') as f:
    pickle.dump(faces, f)

print(f"Total faces collected: {len(faces)}")
print(f"Total labels stored: {len(names)}")
print("Face data saved successfully!")