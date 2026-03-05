from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
import cv2
from app.detector import detect, load_model

app = FastAPI()

@app.on_event("startup")
def startup_event():
    print("Servidor iniciando...")
    load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            data = await websocket.receive_text()

            if "," in data:
                data = data.split(",")[1]

            img_bytes = base64.b64decode(data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Frame inválido ❌")
                continue

            print("Frame recebido:", frame.shape)

            detections = detect(frame)
            await websocket.send_json(detections)

        except Exception as e:
            print("Erro:", e)
            break
