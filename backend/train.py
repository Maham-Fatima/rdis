import cv2
import numpy as np
from PIL import Image
import os

def get_images_and_labels(path, id):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        ID = int(os.path.split(image_path)[-1].split(".")[1])
        if ID >= id:
            img_np = np.array(img, 'uint8')
            detected_faces = face_cascade.detectMultiScale(img_np, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in detected_faces:
                faces.append(img_np[y:y+h, x:x+w])
                ids.append(ID)
    return faces, np.array(ids)

def train(id=1):
    path = 'dataset' 
    faces, ids = get_images_and_labels(path, id)

    # Create the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if os.path.exists('trainer.yml'):
        recognizer.read('trainer.yml')
    # Train the recognizer
    recognizer.train(faces, ids)
    # save model
    recognizer.write('trainer.yml') 
    print("Model trained and saved.")
