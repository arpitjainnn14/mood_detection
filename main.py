from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import cv2
import logging
import datetime
from starlette.middleware.gzip import GZipMiddleware
import asyncio

# Import your modules
from text_emotion import TextEmotionAnalyzer
from audio_emotion import AudioEmotionAnalyzer
from face_detector import FaceDetector
from emotion_analyzer import EmotionAnalyzer
from settings import Settings

# Initialize analyzers (lazy load FaceDetector to avoid heavy startup and fork issues)
text_analyzer = TextEmotionAnalyzer()
audio_analyzer = AudioEmotionAnalyzer()
emotion_analyzer = EmotionAnalyzer()
_face_detector_instance = None

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Load settings
settings = Settings()

# ====== Perf: compression + optional prewarm ======
app.add_middleware(GZipMiddleware, minimum_size=500)

@app.on_event("startup")
async def _maybe_prewarm():
    # Pre-warm transformers to avoid first-request cold start if PREWARM=1
    if os.environ.get("PREWARM", "0") == "1":
        async def warm():
            try:
                # small text to trigger model load
                text_analyzer.analyze_text("hello")
            except Exception:
                pass
        asyncio.create_task(warm())

@app.middleware("http")
async def _static_cache_control(request: Request, call_next):
    response = await call_next(request)
    try:
        path = request.url.path or ""
        if path.startswith("/static/"):
            # Cache static assets aggressively (1 week)
            response.headers["Cache-Control"] = "public, max-age=604800, immutable"
    except Exception:
        pass
    return response

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/text", response_class=HTMLResponse)
async def text_page(request: Request):
    return templates.TemplateResponse("text.html", {"request": request})

@app.post("/api/analyze_text")
async def analyze_text(request: Request):
    try:
        # Handle both form data and JSON
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
            text = data.get("text", "")
        else:
            form_data = await request.form()
            text = form_data.get("text", "")
        
        emotion, confidence = text_analyzer.analyze_text(text)
        
        # Get emoji and description using emotion_analyzer methods
        emoji = emotion_analyzer.get_emotion_emoji(emotion)
        description = emotion_analyzer.get_emotion_description(emotion, confidence)
        
        return JSONResponse(content={
            "emotion": emotion,
            "confidence": confidence,
            "emoji": emoji,
            "description": description
        })
        
    except Exception as e:
        logging.error(f"Text analysis error: {str(e)}")
        return JSONResponse(
            content={"error": f"Analysis failed: {str(e)}"},
            status_code=500
        )

@app.get("/audio", response_class=HTMLResponse)
async def audio_page(request: Request):
    return templates.TemplateResponse("audio.html", {"request": request})

@app.post("/api/analyze_audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        if not os.path.exists("logs"):
            os.makedirs("logs")
        file_path = os.path.join("logs", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        emotion, confidence = audio_analyzer.analyze_audio(file_path)
        
        # Get emoji and description
        emoji = emotion_analyzer.get_emotion_emoji(emotion)
        description = emotion_analyzer.get_emotion_description(emotion, confidence)
        
        return JSONResponse(content={
            "emotion": emotion,
            "confidence": confidence,
            "emoji": emoji,
            "description": description
        })
        
    except Exception as e:
        logging.error(f"Audio analysis error: {str(e)}")
        return JSONResponse(
            content={"error": f"Analysis failed: {str(e)}"},
            status_code=500
        )

@app.get("/vision", response_class=HTMLResponse)
async def vision_page(request: Request):
    return templates.TemplateResponse("vision.html", {"request": request})

@app.post("/api/analyze_frame")
async def analyze_frame(request: Request):
    try:
        # Handle JSON data (base64 image from webcam)
        data = await request.json()
        image_data = data.get("image", "")
        
        if not image_data:
            return JSONResponse(
                content={"error": "No image data provided"},
                status_code=400
            )
        
        # Decode base64 image
        import base64
        import numpy as np
        
        # Remove data URL prefix if present
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                content={"error": "Could not decode image"},
                status_code=400
            )
        
        # Initialize on first use
        global _face_detector_instance
        if _face_detector_instance is None:
            _face_detector_instance = FaceDetector()
        detector = _face_detector_instance
        # Detect faces
        faces, frame_with_faces = detector.detect_faces(image)
        
        if len(faces) == 0:
            return JSONResponse(content={"faces": []})
        
        # Analyze emotion for each detected face
        face_results = []
        for i, face_location in enumerate(faces):
            face_img = detector.extract_face(image, face_location)
            
            if not detector.is_valid_face(face_img):
                continue
            
            # Analyze emotion
            emotion, confidence = emotion_analyzer.analyze_emotion(face_img)
            
            # Get emoji and description
            emoji = emotion_analyzer.get_emotion_emoji(emotion)
            description = emotion_analyzer.get_emotion_description(emotion, confidence)
            
            face_results.append({
                "emotion": emotion,
                "confidence": confidence,
                "emoji": emoji,
                "description": description
            })
        
        return JSONResponse(content={"faces": face_results})
        
    except Exception as e:
        logging.error(f"Frame analysis error: {str(e)}")
        return JSONResponse(
            content={"error": f"Analysis failed: {str(e)}"},
            status_code=500
        )

@app.get("/developers", response_class=HTMLResponse)
async def developers_page(request: Request):
    return templates.TemplateResponse("developers.html", {"request": request})

@app.get("/wellness", response_class=HTMLResponse)
async def wellness_page(request: Request):
    return templates.TemplateResponse("wellness.html", {"request": request})

@app.post("/api/save_screenshot")
async def save_screenshot(file: UploadFile = File(...)):
    try:
        # Create screenshots directory if it doesn't exist
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")
        
        # Generate timestamp filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        filename = f"emotion_{timestamp}{file_extension}"
        file_path = os.path.join("screenshots", filename)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        return JSONResponse(content={"success": True, "filename": filename})
        
    except Exception as e:
        logging.error(f"Screenshot save error: {str(e)}")
        return JSONResponse(
            content={"error": f"Failed to save screenshot: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
