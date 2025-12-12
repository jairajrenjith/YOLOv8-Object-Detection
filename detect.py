from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

img = cv2.imread("images/image.jpg")

results = model(img)[0]

for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    label = f"{model.names[cls]} {conf:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("outputs/results.jpg", img)
