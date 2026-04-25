import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 9: "traffic light"}

cap = cv2.VideoCapture(0)

#hello world