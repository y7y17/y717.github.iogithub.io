import os
import base64
import io
import sys
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add TinyFace package to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinyface import FacePair, TinyFace


app = FastAPI(title="TinyFace API", description="API for TinyFace face swapping", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize TinyFace
tinyface = TinyFace()

# Prepare models on startup
@app.on_event("startup")
async def startup_event():
    print("Loading TinyFace models...")
    tinyface.prepare()
    print("Models loaded successfully!")


class SwapResponse(BaseModel):
    success: bool
    message: str
    swapped_image: Optional[str] = None


# Helper functions
def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image to OpenCV format"""
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    # Decode base64 string
    img_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode OpenCV image to base64 string"""
    # Encode image to jpg format
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Could not encode image")
    
    # Convert to base64
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    
    # Return as data URL
    return f"data:image/jpeg;base64,{img_base64}"


@app.post("/api/swap", response_model=SwapResponse)
async def swap_face(
    source_image: str = Form(...),
    reference_image: str = Form(...),
    destination_image: str = Form(...),
):
    """
    Swap faces in the source image using the reference and destination faces
    
    - source_image: The main image where faces will be swapped (base64)
    - reference_image: The face to be replaced (base64)
    - destination_image: The face to swap in (base64)
    """
    try:
        # Convert base64 images to OpenCV format
        source_img = decode_base64_image(source_image)
        reference_img = decode_base64_image(reference_image)
        destination_img = decode_base64_image(destination_image)
        
        # Get faces from images
        reference_face = tinyface.get_one_face(reference_img)
        if not reference_face:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No face detected in reference image"},
            )
            
        destination_face = tinyface.get_one_face(destination_img)
        if not destination_face:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No face detected in destination image"},
            )
        
        # Swap faces
        output_img = tinyface.swap_face(source_img, reference_face, destination_face)
        
        # Convert result to base64
        output_base64 = encode_image_to_base64(output_img)
        
        return {
            "success": True,
            "message": "Face swap completed successfully",
            "swapped_image": output_base64,
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error: {str(e)}"},
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading TinyFace models...")
    tinyface = TinyFace()
    tinyface.prepare()
    print("Models loaded successfully!")
    yield
@app.post("/api/swap-many", response_model=SwapResponse)
async def swap_many_faces(
    source_image: str = Form(...),
    face_pairs: str = Form(...),  # JSON string with reference and destination face pairs
):
    """
    Swap multiple faces in the source image
    
    - source_image: The main image where faces will be swapped (base64)
    - face_pairs: JSON array of {reference: base64, destination: base64} objects
    """
    try:
        import json
        
        # Convert base64 image to OpenCV format
        source_img = decode_base64_image(source_image)
        
        # Parse face pairs
        pairs_data = json.loads(face_pairs)
        
        # Create face pairs
        face_pairs_list = []
        for pair in pairs_data:
            # Get reference face
            reference_img = decode_base64_image(pair["reference"])
            reference_face = tinyface.get_one_face(reference_img)
            if not reference_face:
                continue
                
            # Get destination face
            destination_img = decode_base64_image(pair["destination"])
            destination_face = tinyface.get_one_face(destination_img)
            if not destination_face:
                continue
                
            # Add to list
            face_pairs_list.append(FacePair(reference=reference_face, destination=destination_face))
        
        if not face_pairs_list:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No valid face pairs detected"},
            )
        
        # Swap faces
        output_img = tinyface.swap_faces(source_img, face_pairs=face_pairs_list)
        
        # Convert result to base64
        output_base64 = encode_image_to_base64(output_img)
        
        return {
            "success": True,
            "message": f"Successfully swapped {len(face_pairs_list)} faces",
            "swapped_image": output_base64,
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error: {str(e)}"},
        )


@app.get("/api/detect-faces", response_model=Dict)
async def detect_faces(image: str = Form(...)):
    """
    Detect faces in an image
    
    - image: The image to detect faces in (base64)
    """
    try:
        # Convert base64 image to OpenCV format
        img = decode_base64_image(image)
        
        # Detect faces
        faces = tinyface.get_many_faces(img)
        
        # Get face bounding boxes
        face_boxes = []
        for face in faces:
            x1, y1, x2, y2 = face.bounding_box
            face_boxes.append({
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "score": float(face.score),
            })
        
        return {
            "success": True, 
            "message": f"Detected {len(faces)} faces",
            "faces": face_boxes
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error: {str(e)}"},
        )


# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)
