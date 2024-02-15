import cv2
import os
import face_recognition
import csv
from datetime import date

video = cv2.VideoCapture(0)

count = 0
password = int(input("enter password"))
# User input for name and address
name = str(input("Enter criminal Name: ")).title()

path = 'images/' + name

if os.path.exists(path):
    print("Name already exists")
    name = str(input("Enter Your Name Again: ")).title()
    path = 'images/' + name
else:
    os.makedirs(path)

# CSV file to store image paths, name, and address

csv_file = 'image_data.csv'
header = ['Image_Path']
if password!=1:
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

while True:
    ret, frame = video.read()
    
    # Convert the frame to RGB (face_recognition uses RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    
    for top, right, bottom, left in face_locations:
        count += 1
        img_name = f"{path}/{count}.jpg"
        print("Creating Images........." + img_name)
        cv2.imwrite(img_name, frame[top:bottom, left:right])
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        
        # Write image path, name, and address to CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ img_name])
    
    cv2.imshow("WindowFrame", frame)
    cv2.waitKey(1)
    
    if count >= 20:
        break

video.release()
cv2.destroyAllWindows()
