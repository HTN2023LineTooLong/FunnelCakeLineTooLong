from ultralytics import YOLO
from PIL import Image
import glob

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Perform object detection on an image using the model

image_list = []
for filename in glob.glob('./img/*.jpg'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)

# Perform detection and capture the print output
results = model.predict(source=image_list, classes=0)

total_people = 0
# Process results list
for result in results:
    boxes = result.boxes
    total_people = total_people + len(boxes)

print(total_people)
