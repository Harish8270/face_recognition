import cv2
import face_recognition
import pandas as pd
import geocoder
import smtplib
from email.message import EmailMessage
from twilio.rest import Client

# Function to get live face encoding
def get_live_face_encoding(frame, top, right, bottom, left):
    # Crop the face from the frame
    face_image = frame[top:bottom, left:right]
    # Convert BGR to RGB (face_recognition uses RGB)
    rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    # Compute face encodings for the face
    face_encodings = face_recognition.face_encodings(rgb_face_image)
    if face_encodings:
        return face_encodings[0]
    return None

# Load dataset from CSV file
csv_file_path = 'image_data.csv'  # Replace 'image_data.csv' with your CSV file path
dataset = pd.read_csv(csv_file_path)

# Initialize webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    # Convert frame to RGB for face_recognition
    rgb_frame = frame[:, :, ::-1]  

    # Find face locations in the live video frame
    live_face_locations = face_recognition.face_locations(rgb_frame)

    for live_face in live_face_locations:
        top, right, bottom, left = live_face
        
        # Get the face encoding for the live face
        live_face_encoding = get_live_face_encoding(frame, top, right, bottom, left)

        if live_face_encoding is not None:
            # Compare face encodings of live frame with dataset images
            for idx, row in dataset.iterrows():
                dataset_image_path = dataset['Image_Path'][idx]
                dataset_image = face_recognition.load_image_file(dataset_image_path)
                dataset_face_encodings = face_recognition.face_encodings(dataset_image)
                
                if dataset_face_encodings:
                    dataset_face_encoding = dataset_face_encodings[0]  # Use the first encoding
                    # Compare the face encodings
                    matches = face_recognition.compare_faces([dataset_face_encoding], live_face_encoding)

                    if matches[0]:
                        print("Match found!")
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
video.release()
cv2.destroyAllWindows()
