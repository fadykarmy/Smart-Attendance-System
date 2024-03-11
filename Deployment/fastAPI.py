# fastAPI.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import dlib
import pickle
from io import BytesIO
from pydantic import BaseModel

app = FastAPI()

# Load known face encodings and names from the trained model file
with open("face_recognition_model.pkl", "rb") as file:
    model_data = pickle.load(file)
    known_face_encodings = model_data['known_face_encodings']
    known_face_names = model_data['known_face_names']

# Initialize the face recognition model
face_recognition_model = dlib.face_recognition_model_v1("shape_predictor_68_face_landmarks.dat")

class ImageInput(BaseModel):
    image: str

def recognize_faces(frame):
    # Resize frame for faster processing (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Find all the faces and face landmarks in the current frame of video
    face_locations = dlib.face_locations(small_frame)
    face_encodings = [face_recognition_model.compute_face_descriptor(small_frame, landmarks) for landmarks in face_locations]

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = dlib.face_distance(known_face_encodings, face_encoding)
        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        best_match_index = np.argmin(matches)
        if matches[best_match_index] < 0.6:  # You may need to adjust the threshold based on your model
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_names

@app.get("/")
async def read_item():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/upload")
async def upload_file(input_data: ImageInput):
    image_data = input_data.image.split(",")[1]  # Remove the data URI prefix
    image_bytes = BytesIO(base64.b64decode(image_data))
    image_np = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)

    # Recognize faces using the loaded model
    recognized_names = recognize_faces(image_np)

    return {
        "recognized_names": recognized_names
    }
