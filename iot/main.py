import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # lightweight model
cap = cv2.VideoCapture("udp://0.0.0.0:5000", cv2.CAP_FFMPEG)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    results = model(frame)
    # draw bounding boxes on frame
    for r in results:
        r.plot()
    cv2.imshow("YOLO Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
