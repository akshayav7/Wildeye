import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import math
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import geocoder
from gtts import gTTS
from playsound import playsound
import os
from collections import deque
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from collections import deque



# ------------------ SESSION STORAGE ------------------

if "detection_history" not in st.session_state:
    st.session_state.detection_history = deque(maxlen=10)

if "already_detected" not in st.session_state:
    st.session_state.already_detected = set()

if "previous_states" not in st.session_state:
    st.session_state.previous_states = {}

if "last_voice_time" not in st.session_state:
    st.session_state.last_voice_time = 0
# ------------------ PAGE NAVIGATION ------------------

# ------------------ CUSTOM NAVIGATION BUTTONS ------------------

if "page" not in st.session_state:
    st.session_state.page = "Detection"

st.sidebar.markdown("## 🚀 Navigation")

# Custom CSS for rounded buttons
st.markdown("""
<style>

/* Sidebar background (deep navy gradient) */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1c, #0e1a2b);
}

/* Make buttons equal width and centered */
div.stButton {
    width: 100%;
}

/* Button style */
div.stButton > button {
    width: 250px;
    height: 50px;
    border-radius: 30px;
    border: none;
    font-size: 16px;
    font-weight: 600;
    color: white;

    /* Gradient matching background */
    background: linear-gradient(135deg, #16222a, #1f2c3a);

    transition: all 0.3s ease;
}

/* Hover effect */
div.stButton > button:hover {
    background: linear-gradient(135deg, #1f2c3a, #243b55);
    transform: scale(1.02);
}

/* Remove blue focus outline */
div.stButton > button:focus {
    outline: none;
    box-shadow: none;
}

</style>
""", unsafe_allow_html=True)

if st.sidebar.button("🐾Detection"):
    st.session_state.page = "Detection"

if st.sidebar.button("📜 History   "):
    st.session_state.page = "Recent History"



page = st.session_state.page

# ------------------ EMAIL CONFIG ------------------

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "armybts87130613@gmail.com"
EMAIL_PASS = "rziv zqpi htrl zpam"
TO_EMAIL = "tanuv0197@gmail.com"

# ------------------ LOCATION ------------------

def get_live_location():
    g = geocoder.ip('me')
    if g.latlng:
        return f"Latitude: {g.latlng[0]}, Longitude: {g.latlng[1]}"
    return "Location not available."

location = get_live_location()

# ------------------ EMAIL FUNCTION ------------------

def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = TO_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)

    except Exception as e:
        print("Email error:", e)

# ------------------ VOICE ALERT ------------------

def play_alarm(animal_name):
    try:
        filename = "alert.mp3"
        tts = gTTS(text=f"{animal_name} detected", lang='en')
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
        # playsound("alert.mp3", False)
    except Exception as e:
        print("Voice error:", e)

# ------------------ MODEL ------------------

model = YOLO('best.pt')

behavior_model = models.resnet18(pretrained=False)
behavior_model.fc = nn.Linear(behavior_model.fc.in_features, 3)

behavior_model.load_state_dict(
    torch.load("animal_behavior_resnet18_final.pth", map_location="cpu")
)

behavior_model.eval()

behavior_classes = ["resting", "walking", "running"]
# ------------------ BEHAVIOR RISK SCORE ------------------

behavior_risk_score = {
    "resting": 0.2,
    "walking": 0.5,
    "running": 0.9
}
running_lock = 0
# ------------------ BEHAVIOR FRAME BUFFER ------------------

behavior_buffer = deque(maxlen=5)
previous_positions = {}
classnames = [
    'antelope','bear','cheetah','human','coyote','crocodile','deer','elephant',
    'flamingo','fox','giraffe','gorilla','hedgehog','hippopotamus','hornbill',
    'horse','hummingbird','hyena','kangaroo','koala','leopard','lion','meerkat',
    'mole','monkey','moose','okapi','orangutan','ostrich','otter','panda',
    'pelecaniformes','porcupine','raccoon','reindeer','rhino','rhinoceros',
    'snake','squirrel','swan','tiger','turkey','wolf','woodpecker','zebra',
    'wild boar','bison','buffalo','panther','jaguar','puma','elephant seal',
    'grizzly bear','polar bear','giant panda','red panda','chimpanzee',
    'snow leopard','sea lion','walrus','manatee','wolverine','wild dog',
    'serval','jackal','camel','albatross','vulture','eagle','falcon','owl',
    'hawk','kite','penguin','seal','whale','orca','dolphin','shark','ray',
    'stingray','cuttlefish','octopus'
]

HIGH_RISK = {
    "lion","tiger","leopard","cheetah","bear","grizzly bear","polar bear",
    "jaguar","panther","wolf","wild dog","hyena","crocodile","snake",
    "elephant","gorilla","orca","hippopotamus","buffalo","bison","wild boar"
}

MEDIUM_RISK = {
    "rhino","rhinoceros","wolverine","serval","jackal","coyote","moose","camel"
}

# ------------------ SPECIES RISK SCORE ------------------

species_risk_score = {
    "HIGH": 0.9,
    "MEDIUM": 0.6,
    "LOW": 0.3
}
wild_animals = set(classnames)
human_class = "human"

# ------------------ DETECTION FUNCTION ------------------

def detect(frame):
    result = model(frame, stream=True)
    detected_animals = []
    detected_behaviors = []

    for info in result:
        for box in info.boxes:
            confidence = int(box.conf[0] * 100)
            class_index = int(box.cls[0])

            if confidence > 50:
                label = classnames[class_index]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                speed = 0

                if label in previous_positions:
                    prev_x, prev_y = previous_positions[label]
                    speed = math.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)

                previous_positions[label] = (center_x, center_y)
                # Crop the detected animal
                animal_crop = frame[y1:y2, x1:x2]

                behavior = "Analyzing..."
                # If movement is very fast → classify as running
                # if speed > 2:
                #     behavior = "running"

                if animal_crop.size != 0:

                    behavior_buffer.append(animal_crop)

                    if len(behavior_buffer) == 5:

                        predictions = []

                        for crop in behavior_buffer:
                            pred = predict_behavior(crop)
                            predictions.append(pred)

                        # majority vote
                        behavior = max(set(predictions), key=predictions.count)
                        global running_lock

                        if speed > 20:
                            behavior = "running"
                            running_lock = 10   # keep running for next 10 frames

                        elif running_lock > 0:
                            behavior = "running"
                            running_lock -= 1

                # ------------------ HUMAN DETECTION ------------------
                if label == human_class:

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                    cv2.putText(frame, f'{label} {confidence}%',
                                (x1,y1-25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,(255,255,255),2)

                    # Add human to detected list
                    detected_animals.append("human")

                # ------------------ WILD ANIMAL DETECTION ------------------
                elif label in wild_animals:

                    if label in HIGH_RISK:
                        threat = "HIGH"
                        color = (0,0,255)
                    elif label in MEDIUM_RISK:
                        threat = "MEDIUM"
                        color = (0,255,255)
                    else:
                        threat = "LOW"
                        color = (0,255,0)

                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)

                    
                    cv2.putText(frame, f'{label} ({behavior}) {confidence}%',
                                (x1,y1-25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,(255,255,255),2)
                    # ------------------ THREAT SCORE CALCULATION ------------------

                    species_score = species_risk_score.get(threat, 0.3)

                    behavior_score = behavior_risk_score.get(behavior, 0.2)

                    threat_score = (0.7 * species_score) + (0.3 * behavior_score)

                    # Decide final threat level based on score

                    if threat_score >= 0.7:
                        threat = "HIGH"
                        color = (0,0,255)

                    elif threat_score >= 0.45:
                        threat = "MEDIUM"
                        color = (0,255,255)

                    else:
                        threat = "LOW"
                        color = (0,255,0)

                    cv2.putText(frame, f'Threat: {threat}',
                                (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,color,2)
                    cv2.putText(frame, f'Score: {threat_score:.2f}',
                                (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, color, 2)
                    if label not in st.session_state.already_detected:

                        st.session_state.already_detected.add(label)

                        filename = f"detection_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)

                        st.session_state.detection_history.appendleft({
                            "Animal": label,
                            "Threat": threat,
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Location": location,
                            "Image": filename
                        })

                        threading.Thread(
                            target=send_email,
                            args=(
                                f"Wild Animal Detected: {label}",
                                f"{label} detected at {datetime.now()}\nLocation: {location}"
                            )
                        ).start()

                    detected_animals.append(label)
                    detected_behaviors.append(behavior)

    return frame, detected_animals,detected_behaviors

def predict_behavior(crop):

    crop = cv2.resize(crop, (224,224))

    crop = crop / 255.0

    crop = np.transpose(crop, (2,0,1))  # HWC → CHW

    crop = torch.tensor(crop, dtype=torch.float32).unsqueeze(0)

    # ImageNet normalization
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)

    crop = (crop - mean) / std

    with torch.no_grad():
        output = behavior_model(crop)
        pred = torch.argmax(output, dim=1).item()

    return behavior_classes[pred]

# ------------------ DETECTION PAGE ------------------


if page == "Detection":

    st.title("🐾 Wildlife Intrusion Detection")

    option = st.selectbox(
        "Choose Input Type",
        ["Upload Image", "Upload Video"]
    )

    # ------------------ IMAGE ------------------

    if option == "Upload Image":

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:

            file_bytes = np.asarray(
                bytearray(uploaded_file.read()),
                dtype=np.uint8
            )

            image = cv2.imdecode(file_bytes, 1)

            frame, detected_animals,detected_behaviors = detect(image)

            st.image(frame, channels="BGR")

            if detected_animals:
                for animal in detected_animals:
                    threading.Thread(
                        target=play_alarm,
                        args=(animal,)
                    ).start()

                    

                st.warning(f"Detected: {', '.join(detected_animals)}")

    # ------------------ VIDEO ------------------

    elif option == "Upload Video":

        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"]
        )

        if uploaded_file is not None:

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            vid_cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            frame_count = 0

            while vid_cap.isOpened():

                ret, frame = vid_cap.read()

                if not ret:
                    break

                frame, detected_animals ,detected_behaviors = detect(frame)

                stframe.image(frame, channels="BGR")

                if detected_animals:
                    for animal, behavior in zip(detected_animals, detected_behaviors):

                        previous = st.session_state.previous_states.get(animal)

                        # trigger only if new animal or behavior changed
                        if previous != behavior:

                            st.session_state.previous_states[animal] = behavior
                            


                            st.warning(f"{animal} detected ({behavior})")
                            current_time = time.time()
                            if current_time - st.session_state.last_voice_time > 3:
                                st.session_state.last_voice_time = current_time

                            threading.Thread(
                                target=play_alarm,
                                args=(animal,)
                            ).start()

                        

            vid_cap.release()

    # ------------------ WEBCAM ------------------

    # ------------------ WEBCAM ------------------

    elif option == "Open Webcam":

        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                st.error("Failed to open webcam. Check permissions!")
                break

            frame, detected_animals,detected_behaviors = detect(frame)

            stframe.image(frame, channels="BGR")

            # Check if human detected
            if "human" in detected_animals:
                st.success("Human detected via webcam!")

            # Handle animal detection
            animal_list = [a for a in detected_animals if a != "human"]

            if animal_list:
                for animal in animal_list:

                    

                    threading.Thread(
                        target=send_email,
                        args=(
                            f"Wild Animal Detected: {animal}",
                            f"A wild animal, specifically a {animal}, was detected via the webcam at {datetime.now()}.\nLocation: {location}"
                        )
                    ).start()

        cap.release()
# ------------------ HISTORY PAGE ------------------

if page == "Recent History":

    st.title("📜 Detection History")

    if not st.session_state.detection_history:
        st.info("No detections yet.")
    else:
        for entry in st.session_state.detection_history:
            with st.container():
                col1, col2 = st.columns([1,2])

                with col1:
                    st.image(entry["Image"], width=300)

                with col2:
                    st.markdown(f"### 🐾 {entry['Animal']}")
                    st.write(f"**Threat Level:** {entry['Threat']}")
                    st.write(f"**Time:** {entry['Time']}")
                    st.write(f"**Location:** {entry['Location']}")

                st.markdown("---")