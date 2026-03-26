# 🐾 WildEye – Smart Animal Intrusion Detection & Behavior Monitoring System

An intelligent real-time wildlife detection and behavior analysis system designed to prevent human-wildlife conflicts.  
WildEye uses deep learning models to detect animals, analyze their behavior, and trigger alerts through alarms and email notifications.

---

## 📖 Overview

WildEye is a smart surveillance system that monitors video or image input to detect wild animals and analyze their behavior.  
The system combines object detection and behavior classification to assess potential threats and alert users instantly.

When an animal is detected, the system:
- Identifies the animal using YOLOv8
- Analyzes behavior using a trained ResNet-18 model
- Calculates threat level based on detection + behavior
- Triggers alerts (alarm + email)

---

## 🚀 Key Features

- 🔍 **Real-Time Animal Detection** using YOLOv8  
- 🧠 **Behavior Analysis Module** (ResNet-18 based model)  
- ⚠️ **Threat Level Calculation** based on species + behavior  
- 🌐 **Streamlit Web Interface** for user interaction  
- 🚨 **Sound Alarm System** for immediate alerts  
- 📧 **Email Notifications** for remote alerting  
- 🎥 **Supports Image & Video Input**  
- 🐍 Built fully using Python  

---

## 🧠 Models Used

- **YOLOv8 (.pt file)** → Detects animals in frames  
- **ResNet-18 (.pth file)** → Classifies animal behavior  

---

## 🏗️ Project Structure
wildlife_detection/
│
├── animal.py # Main Streamlit application
├── alarm.py # Alarm triggering logic
├── best.pt # YOLOv8 detection model
├── animal_behavior_resnet18_final.pth # Behavior model
├── alarm.mp3 # Alarm sound
├── requirements.txt # Dependencies
├── video/ # Sample videos
└── README.md

## ⚙️ How It Works

1. Input video/image is given through Streamlit UI  
2. YOLOv8 detects animals in frames  
3. Detected animal is passed to behavior model  
4. Behavior is classified (e.g., calm, aggressive, etc.)  
5. Threat score is calculated  
6. If threat is high:
   - Alarm is triggered 🔊  
   - Email is sent 📧  

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- OpenCV  
- PyTorch  
- YOLOv8  
- ResNet-18  
- SMTP (Email alerts)
