import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

#defining what to detect
TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 9: "traffic light"}

PRIORITY = {0: 1, 2: 2, 3: 2, 9: 3} #priority of detected objects, lower number means higher priority, person has highest priority, traffic light has lowest priority, cae and motorcycle have same priority

def estimate_distance(box_height, frame_height):
    ratio = box_height / frame_height
    if ratio > 0.5:
        return "VERY CLOSE"
    elif ratio > 0.3:
        return "CLOSE"
    elif ratio > 0.1:
        return "FAR"
    else:
        return "VERY FAR"
    
def get_priority_label(cls_id):
    p = PRIORITY[cls_id]
    if p == 1:
        return "HIGH PRIORITY"
    elif p == 2:
        return "MEDIUM PRIORITY"
    else:
        return "LOW PRIORITY"
    
def calculate_risk(cls_id, distance):
    if cls_id == 0: #person
        if distance == "VERY CLOSE":
            return 10
        elif distance == "CLOSE":
            return 7
        elif distance == "FAR":
            return 3
        else:
            return 1
    elif cls_id == 2 or cls_id == 3: #car or motorcycle
        if distance == "VERY CLOSE":
            return 7
        elif distance == "CLOSE":
            return 5
        elif distance == "FAR":
            return 2
        else:
            return 1
    else: #traffic light
        return 0

def make_decision(total_risk):
    if total_risk >= 7:
        return "STOP", (0, 0, 255) #red color for stop
    elif total_risk >= 4:
        return "SLOW DOWN", (0, 165, 255) #orange color for slow down
    else:
        return "GO", (0, 255, 0) #green color for go
    
def get_explanation(decision, detected_objects):
    if not detected_objects:
        return "road is clear, safe to continue"
    top = sorted(detected_objects, key=lambda x: x[0])[0] #get the highest priority detected object
    cls_id, distance = top
    
    total = len(detected_objects)
    multi = f"{total} obstacles detected. " if total > 1 else ""

    if decision == "STOP":
        if cls_id == 0:
            return f"{multi}Pedestrian at a {distance.lower()} range. Stop immediately."
        elif cls_id == 2 or cls_id == 3:
            return f"{multi}Vehicle at a {distance.lower()} range. Stop immediately."
        else:
            return f"{multi}Risk detected. You should stop"
        
    if decision == "SLOW DOWN":
        if cls_id == 0:
            return f"{multi}Pedestrian at a {distance.lower()} range spotted. Slow down."
        elif cls_id == 2 or cls_id == 3:
            return f"{multi}Vehicle at a {distance.lower()} range spotted. Be careful, reduce speed."
        else:
            return f"{multi}Risk detected. Proceed with caution."
        
    else:
        return f"{multi}No risk detected. road is clear, safe to continue."
        


#initializing webcam, 0 = first webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():#while camera is open
    ret, frame = cap.read() #ret = boolian(true/false), frame = image
    if not ret: #if there is no frame,
        break

    frame_height = frame.shape[0] #get the height of the frame to estimate distance of detected objects
    frame_width = frame.shape[1] 
    #run the model on the frame, verbose = False to disable printing results to console, [0] to get the first result (since model can return multiple results for batch processing, we dont need that here since we are processing one frame at a time) 
    results = model(frame, verbose=False)[0] 

    detections = [] #list to store detected objects and their priorities
    for box in results.boxes: #box is variable, and results.boxes predefined in YOLO model to get the bounding boxes of detected objects
        cls_id = int(box.cls[0]) #get the class id of the detected object, [0] to get the first extract from tensor
        if cls_id not in TARGET_CLASSES: #if the class id is not in our target classes, skip it
            continue
        detections.append((PRIORITY[cls_id], box, cls_id))#append/add the priority and the box and id to the detections list
    
    detections.sort(key=lambda x: x[0])#sort the detections list based on priority, lower number means higher priority

    total_risk = 0 #variable to store the total risk of the detected objects in the frame
    detected_objects = [] #list to store the detected objects and their distances for explanation

    for _, box, cls_id in detections: #iterate through the sorted detections list, _ is used to ignore the priority value since we dont need it for drawing the bounding box and label
        label = TARGET_CLASSES[cls_id] #get the label of the detected object using the class id
        conf = float(box.conf[0]) #get the confidence of the detected object, [0] to extract the first confidence from tensor 
        x1, y1, x2, y2 = map(int, box.xyxy[0]) #get the coordinates of the bounding box, [0] to extract the first box coordinates from tensor

        box_height = y2 - y1 #calculate the height of the bounding box to estimate distance of detected object
        distance = estimate_distance(box_height, frame_height) #estimate the distance of the detected object
        priority = get_priority_label(cls_id) #get the priority label of the detected object

        risk = calculate_risk(cls_id, distance) #calculate the risk of the detected object based on its class and distance
        total_risk += risk #add the risk of the detected object to the total risk

        if cls_id == 0:
            color = (0, 0, 255) #red color for person
        elif cls_id == 2 or cls_id == 3:
            color = (0, 165, 255) #orange color for car and motorcycle
        else:
            color = (0, 255, 255) #yellow color for traffic light

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f} | {distance} | {priority}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    decision, decision_color = make_decision(total_risk) #make a decision based on the total risk of the detected objects in the frame

    cv2.rectangle(frame, (0, 0), (frame_width, 60), (0, 0, 0), -1)  # black bar
    cv2.putText(frame, f"DECISION: {decision}  |  RISK: {total_risk}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, decision_color, 3)

    cv2.imshow("smart driving", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #0xFF is a bitwise operation to get the last 8 bits of the key pressed, ord('q') gets the ASCII value of 'q'
        break

cap.release() 
cv2.destroyAllWindows() 



