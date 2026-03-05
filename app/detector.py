from ultralytics import YOLO

model = None

def load_model():
    global model
    model = YOLO("yolov26n.pt")
    print("Modelo carregado!")

def detect(frame):
    results = model(frame, conf=0.10)

    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "label": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "xyxy": box.xyxy.tolist()
            })

    return detections
