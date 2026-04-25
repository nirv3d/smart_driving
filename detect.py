import cv2
from ultralytics import YOLO

#loading the model
model = YOLO("yolov8n.pt")

#defining what to detect
TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 9: "traffic light"}

#opening webcam, 0 = first webcam
cap = cv2.VideoCapture(0)

