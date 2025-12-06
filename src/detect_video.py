from ultralytics import YOLO
import cv2
from count_fingers import count_fingers

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)[0]

    if results.keypoints:
        kps = results.keypoints[0].xy.squeeze().tolist()
        fingers_up = count_fingers(kps)
        cv2.putText(frame, f"Dedos: {fingers_up}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Pose", frame)
    if cv2.waitKey(1) == 27:
        break
