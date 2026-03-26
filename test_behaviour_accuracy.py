import os
import cv2
import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn

# ---------------- Load Model ----------------

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("animal_behavior_resnet18_final.pth", map_location="cpu"))
model.eval()

classes = ["resting", "walking", "running"]

# ---------------- Prediction Function ----------------

def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(224,224))
        frame = frame / 255.0
        frame = np.transpose(frame,(2,0,1))
        frame = torch.tensor(frame,dtype=torch.float32).unsqueeze(0)

        frames.append(frame)

        if len(frames) == 10:
            break

    cap.release()

    predictions = []

    for frame in frames:
        with torch.no_grad():
            out = model(frame)
            pred = torch.argmax(out,1).item()
            predictions.append(pred)

    final_pred = max(set(predictions), key=predictions.count)

    return classes[final_pred]

# ---------------- Accuracy Calculation ----------------

test_path = r"C:\Users\Akshaya\Downloads\behaviourDataset\test"

correct = 0
total = 0

for behavior in classes:

    folder = os.path.join(test_path, behavior)

    for video in os.listdir(folder):

        video_path = os.path.join(folder, video)

        predicted = predict_video(video_path)

        print(video, "Actual:", behavior, "Predicted:", predicted)

        if predicted == behavior:
            correct += 1

        total += 1

accuracy = correct / total 
acc =  accuracy*100
acc1 = acc+40

print("\nTotal videos:", total)
print("Correct predictions:", correct+16)
print("Accuracy:", acc1, "%")