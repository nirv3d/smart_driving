# 🚗 Smart Driving Decision System  
**YOLOv8 · OpenCV · Flask · FastAPI · Streamlit**

A real-time AI-powered driving assistant that detects road objects, estimates risk, and outputs human-readable driving decisions like **STOP / SLOW DOWN / GO**.

---

## 🔍 Overview  
This system combines **computer vision and backend engineering** to simulate intelligent driving decisions in real time. It processes live video streams, detects objects, evaluates proximity, and generates actionable insights with explanations.

---

## ⚙️ Key Features  
-  **Object Detection** using YOLOv8 (cars, pedestrians, traffic lights, etc.)  
-  **Proximity Estimation** via bounding box heuristics  
-  **Decision Engine** with real-time risk scoring  
-  **Natural Language Explanations** for every decision  
-  **Live Video Streaming** (MJPEG via Flask)  
-  **REST API Support** using FastAPI (microservice architecture)  
-  **Interactive Dashboard** with visual overlays and risk indicators  

---

## 🧱 System Architecture  
**4-Phase ML Pipeline:**
Object Detection → Context Understanding → Decision Engine → Explanation Generator

---

## 🚀 Use Cases  
- Driver assistance systems  
- Autonomous vehicle prototyping  
- Traffic monitoring & safety analytics  
- AI-based decision systems  

---

## 📌 Highlights  
- Real-time performance with smooth video inference  
- Modular microservice-based architecture  
- Scalable API design for external integrations  

