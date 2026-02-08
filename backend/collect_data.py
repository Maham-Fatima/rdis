import os
import cv2
import numpy as np
from train import train

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def get_max_id(folder='dataset'):
    max_id = 0
    for f in os.listdir(folder):
        filename = os.path.basenamqe(f)
        parts = filename.split(".")
        if len(parts) >= 3:
            id_num = int(parts[1])
            if id_num > max_id:
                max_id = id_num
    return max_id

def collect_data():
    found = False
    max_id = get_max_id()
    counter = 0
    MAX_IMAGES = 50
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, img = capture.read()
        if not ret:
            continue
        if counter >= MAX_IMAGES:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            found = True
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_img = gray[y:y+h, x:x+w]
            file_index = len(os.listdir('dataset'))
            file_path = f"dataset/user.{max_id+1}.{file_index}.jpg"
            cv2.imwrite(file_path, face_img)
            counter+=1

        
        resized = cv2.resize(img, (500, 500))
        cv2.imshow("Face Capture", resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    if found:
        train(max_id+1)

if __name__ == "__main__":
    collect_data()
