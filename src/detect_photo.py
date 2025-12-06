from ultralytics import YOLO
import cv2
from count_fingers import count_fingers

model = YOLO("best.pt")

img = cv2.imread("input.jpg")
results = model(img)[0]

if results.keypoints:
    kps = results.keypoints[0].xy.squeeze().tolist()
    fingers_up = count_fingers(kps)
    print("Dedos levantados:", fingers_up)

cv2.imshow("resultado", img)
cv2.waitKey(0)
