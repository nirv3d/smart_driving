import cv2
import time
import base64
import threading
from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8n.pt")

TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 9: "traffic light"}
PRIORITY = {0: 1, 2: 2, 3: 2, 9: 3}

# ── Shared State ──────────────────────────────────────────────
state = {
    "decision": "GO",
    "risk": 0,
    "reason": "Analyzing...",
    "detections": [],
    "frame_b64": "" #will hold the latest frame in base64 format to send over API
}

# ── Functions ─────────────────────────────────────────────────
def estimate_distance(box_height, frame_height):
    ratio = box_height / frame_height
    if ratio > 0.5:   return "VERY NEAR"
    elif ratio > 0.3: return "NEAR"
    elif ratio > 0.1: return "FAR"
    else:             return "VERY FAR"

def calculate_risk(cls_id, distance):
    if cls_id == 0:
        if distance == "VERY NEAR": return 10
        elif distance == "NEAR":    return 7
        elif distance == "FAR":     return 3
        else:                       return 1
    elif cls_id in [2, 3]:
        if distance == "VERY NEAR": return 7
        elif distance == "NEAR":    return 5
        elif distance == "FAR":     return 2
        else:                       return 1
    return 0

def make_decision(total_risk):
    if total_risk >= 7:   return "STOP"
    elif total_risk >= 4: return "SLOW DOWN"
    else:                 return "GO"

def get_explanation(decision, detected_objects):
    if not detected_objects:
        return "Road is clear, safe to continue."
    top = sorted(detected_objects, key=lambda x: PRIORITY[x[0]])[0]
    cls_id, distance = top
    total = len(detected_objects)
    multi = f"{total} obstacles detected. " if total > 1 else ""
    if decision == "STOP":
        if cls_id == 0:        return f"{multi}Pedestrian at {distance.lower()} range, stopping immediately."
        elif cls_id in [2, 3]: return f"{multi}Vehicle too close, stopping for safety."
        else:                  return f"{multi}High risk ahead, stopping immediately."
    elif decision == "SLOW DOWN":
        if cls_id == 0:        return f"{multi}Pedestrian spotted ahead, reducing speed."
        elif cls_id in [2, 3]: return f"{multi}Vehicle at {distance.lower()} range, slowing down."
        else:                  return f"{multi}Object detected ahead, proceed with caution."
    return "No obstacles detected, road is clear."

# ── Background Detection Thread ───────────────────────────────
def run_detection(): 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    last_explanation_time = 0
    last_decision = ""
    explanation = "Analyzing..."

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))  # add this line to resize the frame to a smaller size for faster processing
        if not ret:
            continue

        frame_height = frame.shape[0]
        results = model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TARGET_CLASSES:
                continue
            detections.append((PRIORITY[cls_id], box, cls_id))

        detections.sort(key=lambda x: x[0])

        total_risk = 0
        detected_objects = []
        detection_list = []

        for _, box, cls_id in detections:
            label = TARGET_CLASSES[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_height = y2 - y1
            distance = estimate_distance(box_height, frame_height)
            risk = calculate_risk(cls_id, distance)
            total_risk += risk
            detected_objects.append((cls_id, distance))
            detection_list.append({
                "label": label,
                "confidence": round(conf, 2),
                "distance": distance,
                "risk": risk
            })

            if cls_id == 0:        color = (0, 0, 255)
            elif cls_id in [2, 3]: color = (0, 165, 255)
            else:                  color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} | {distance}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        decision = make_decision(total_risk)

        current_time = time.time()
        if decision != last_decision or (current_time - last_explanation_time) > 3:
            explanation = get_explanation(decision, detected_objects)
            last_explanation_time = current_time
            last_decision = decision

        # Convert frame to base64 to send over API
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50]) # reduce quality for smaller size
        frame_b64 = base64.b64encode(buffer).decode("utf-8")

        # Update shared state
        state["decision"] = decision
        state["risk"] = total_risk
        state["reason"] = explanation
        state["detections"] = detection_list
        state["frame_b64"] = frame_b64

        time.sleep(0.03)

# ── Start Detection Thread ────────────────────────────────────
thread = threading.Thread(target=run_detection, daemon=True)
thread.start()

# ── API Endpoints ─────────────────────────────────────────────
@app.get("/decision")
def get_decision():
    return {
        "decision": state["decision"],
        "risk": state["risk"],
        "reason": state["reason"]
    }

@app.get("/detections")
def get_detections():
    return {"detections": state["detections"]}

@app.get("/frame")
def get_frame():
    return {"frame": state["frame_b64"]}