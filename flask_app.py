import cv2
import time
import threading
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8n.pt")

TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 9: "traffic light"}
PRIORITY = {0: 1, 2: 2, 3: 2, 9: 3}

state = {
    "decision": "GO",
    "risk": 0,
    "reason": "Analyzing road...",
    "detections": [],
    "decision_color": "#00cc66"
}
state_lock = threading.Lock()

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
    if total_risk >= 7:   return "STOP", "#ff4444"
    elif total_risk >= 4: return "SLOW DOWN", "#ffaa00"
    else:                 return "GO", "#00cc66"

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

def detection_thread():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    last_explanation_time = 0
    last_decision = ""
    explanation = "Analyzing road..."
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        frame_height = frame.shape[0]

        total_risk = 0
        detected_objects = []
        detection_list = []

        results = model(frame, verbose=False, conf=0.3)[0]
        detections = []

        for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in TARGET_CLASSES:
                    continue
                detections.append((PRIORITY[cls_id], box, cls_id))

        detections.sort(key=lambda x: x[0])

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
                    "confidence": f"{conf:.2f}",
                    "distance": distance,
                    "risk": risk
                })

                if cls_id == 0:        color = (0, 0, 255)
                elif cls_id in [2, 3]: color = (0, 165, 255)
                else:                  color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} | {distance}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        decision, decision_color = make_decision(total_risk)

        current_time = time.time()
        if decision != last_decision or (current_time - last_explanation_time) > 3:
            explanation = get_explanation(decision, detected_objects)
            last_explanation_time = current_time
            last_decision = decision

        # Draw decision bar on frame
        cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), -1)
        if decision == "STOP":         bar_color = (0, 0, 255)
        elif decision == "SLOW DOWN":  bar_color = (0, 165, 255)
        else:                          bar_color = (0, 255, 0)
        cv2.putText(frame, f"{decision} | RISK: {total_risk}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bar_color, 2)

        # Update shared state with lock
        with state_lock:
            state["decision"] = decision
            state["risk"] = total_risk
            state["reason"] = explanation
            state["detections"] = detection_list
            state["decision_color"] = decision_color
            state["frame"] = frame.copy()

def generate_frames():
    while True:
        with state_lock:
            frame = state.get("frame", None)

        if frame is None:
            time.sleep(0.03)
            continue

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        time.sleep(0.03)

thread = threading.Thread(target=detection_thread, daemon=True)
thread.start()

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/data")
def data():
    with state_lock:
        return jsonify({
            "decision": state["decision"],
            "risk": state["risk"],
            "reason": state["reason"],
            "detections": state["detections"],
            "decision_color": state["decision_color"]
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)