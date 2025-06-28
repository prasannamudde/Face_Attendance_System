

import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, db
import numpy as np

# Initialize Firebase using your saved key
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://face-recognition-and-firebase-default-rtdb.firebaseio.com/'
})

# Get student details
name = input("Enter student name: ")
roll = input("Enter roll number: ")
email = input("Enter email: ")

# Start webcam
cap = cv2.VideoCapture(0)
print("📸 Press SPACE to capture face...")

face_encoding = None

while True:
    success, frame = cap.read()
    cv2.imshow("Register - Press SPACE", frame)
    key = cv2.waitKey(1)

    if key % 256 == 32:  # SPACE pressed
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        if encodings:
            face_encoding = encodings[0]
            print("✅ Face captured and encoded.")
        else:
            print("❌ No face found. Try again.")
        break

cap.release()
cv2.destroyAllWindows()

# Upload to Firebase
if face_encoding is not None:
    ref = db.reference("students")
    ref.push({
        "name": name,
        "roll": roll,
        "email": email,
        "encoding": face_encoding.tolist()
    })
    print("✅ Student data uploaded to Firebase!")
else:
    print("❌ Registration failed. Face not detected.")
