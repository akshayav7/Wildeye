import os
import cv2
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# Load model
model = YOLO("best.pt")

TEST_FOLDER = r"C:\Users\Akshaya\Downloads\wildeye_test"

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

y_true = []
y_pred = []

for file in os.listdir(TEST_FOLDER):
    if file.endswith((".jpg", ".png", ".jpeg")):

        actual_label = ''.join([c for c in file.split('.')[0] if not c.isdigit()]).lower()

        img_path = os.path.join(TEST_FOLDER, file)
        image = cv2.imread(img_path)

        results = model(image)

        predicted_label = "none"

        for r in results:
            if len(r.boxes) > 0:
                cls_id = int(r.boxes[0].cls[0])
                predicted_label = classnames[cls_id]
                break

        y_true.append(actual_label)
        y_pred.append(predicted_label)

        print(f"{file} → Actual: {actual_label}, Predicted: {predicted_label}")

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred))

print("\n📊 Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

accuracy = accuracy_score(y_true, y_pred)

print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")