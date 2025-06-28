

import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, db
import numpy as np
from datetime import datetime

# Initialize Firebase using your saved key
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://face-recognition-and-firebase-default-rtdb.firebaseio.com/'
})

# Load all registered students from Firebase
students_ref = db.reference("students")
students = students_ref.get()

known_encodings = []
student_data = []

if students:
    for key, value in students.items():
        known_encodings.append(np.array(value['encoding']))
        student_data.append({
            "id": key,
            "name": value['name'],
            "roll": value['roll'],
            "email": value['email']
        })

# Start webcam for attendance
cap = cv2.VideoCapture(0)
marked = set()

print("🟢 Starting real-time attendance. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb_small)
    encodings = face_recognition.face_encodings(rgb_small, faces)

    for encoding, location in zip(encodings, faces):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        distances = face_recognition.face_distance(known_encodings, encoding)

        best_match = np.argmin(distances)
        if matches[best_match]:
            name = student_data[best_match]['name']
            roll = student_data[best_match]['roll']
            student_id = student_data[best_match]['id']

            if name not in marked:
                # Mark attendance in Firebase
                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                attendance_ref = db.reference("attendance")
                attendance_ref.push({
                    "student_id": student_id,
                    "name": name,
                    "roll": roll,
                    "date": date,
                    "time": time,
                    "status": "Present"
                })

                marked.add(name)
                print(f"✅ {name} marked present at {time}")

            # Draw box and label
            top, right, bottom, left = [v * 4 for v in location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Real-Time Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
