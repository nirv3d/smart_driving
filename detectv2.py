import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

#defining what to detect
TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 9: "traffic light"}

#initializing webcam, 0 = first webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():#while camera is open
    ret, frame = cap.read() #ret = boolian(true/false), frame = image
    if not ret: #if there is no frame,
        break

    #run the model on the frame, verbose = False to disable printing results to console, [0] to get the first result (since model can return multiple results for batch processing, we dont need that here since we are processing one frame at a time) 
    results = model(frame, verbose=False)[0] 

    for box in results.boxes: #box is variable, and results.boxes predefined in YOLO model to get the bounding boxes of detected objects
        cls_id = int(box.cls[0]) #get the class id of the detected object, [0] to get the first extract from tensor
        if cls_id not in TARGET_CLASSES: #if the class id is not in our target classes, skip it
            continue 

        label = TARGET_CLASSES[cls_id] #get the label of the detected object using the class id
        conf = float(box.conf[0]) #get the confidence of the detected object, [0] to extract the first confidence from tensor 
        x1, y1, x2, y2 = map(int, box.xyxy[0]) #get the coordinates of the bounding box, [0] to extract the first box coordinates from tensor

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Smart Driving", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #0xFF is a bitwise operation to get the last 8 bits of the key pressed, ord('q') gets the ASCII value of 'q'
        break

cap.release() 
cv2.destroyAllWindows() 



