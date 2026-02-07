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
        id_num = os.path.split()[-1].split(".")[1]
        if id_num > max_id:
            max_id = id_num
    return max_id

def collect_data():
    max_id = get_max_id()
    capture = cv2.VideoCapture(0)

    while True:
        ret, img = capture.read()
        if not ret:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_img = gray[y:y+h, x:x+w]
            file_index = len(os.listdir('dataset'))
            file_path = f"dataset/user.{max_id+1}.{file_index}.jpg"
            cv2.imwrite(file_path, face_img)

        cv2.imshow("Face Capture", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

    train(max_id+1)
