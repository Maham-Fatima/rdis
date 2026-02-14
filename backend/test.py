import cv2

cap = cv2.VideoCapture("http://192.168.100.52:8080/video")

while True:
    ret, img = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
