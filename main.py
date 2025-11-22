from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import logging
import datetime
from starlette.middleware.gzip import GZipMiddleware
import asyncio

# Lazy imports for better startup performance
_cv2 = None
_text_analyzer = None
_audio_analyzer = None
_emotion_analyzer = None
_face_detector_instance = None
_settings = None

def get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2

def get_text_analyzer():
    global _text_analyzer
    if _text_analyzer is None:
        from text_emotion import TextEmotionAnalyzer
        _text_analyzer = TextEmotionAnalyzer()
        logging.info("Text analyzer initialized")
    return _text_analyzer

def get_audio_analyzer():
    global _audio_analyzer
    if _audio_analyzer is None:
        from audio_emotion import AudioEmotionAnalyzer
        _audio_analyzer = AudioEmotionAnalyzer()
        logging.info("Audio analyzer initialized")
    return _audio_analyzer

def get_emotion_analyzer():
    global _emotion_analyzer
    if _emotion_analyzer is None:
        from emotion_analyzer import EmotionAnalyzer
        _emotion_analyzer = EmotionAnalyzer()
        logging.info("Emotion analyzer initialized")
    return _emotion_analyzer

def get_face_detector():
    global _face_detector_instance
    if _face_detector_instance is None:
        from face_detector import FaceDetector
        _face_detector_instance = FaceDetector()
        logging.info("Face detector initialized")
    return _face_detector_instance

def get_settings():
    global _settings
    if _settings is None:
        from settings import Settings
        _settings = Settings()
    return _settings

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Settings will be loaded lazily

# ====== Perf: compression + optional prewarm ======
app.add_middleware(GZipMiddleware, minimum_size=500)

@app.on_event("startup")
async def _startup():
    # Minimal startup - just set OpenCV threads if specified
    try:
        threads_str = os.environ.get("OPENCV_THREADS", "2")  # Default to 2 threads
        if threads_str:
            os.environ["OPENCV_NUM_THREADS"] = threads_str
    except Exception:
        pass
    
    logging.info("TheraVox server started - models will be loaded on first use")

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
        
        # Offload CPU-bound analysis to a worker thread to keep event loop snappy
        loop = asyncio.get_event_loop()
        
        def analyze_text_sync():
            analyzer = get_text_analyzer()
            return analyzer.analyze_text(text)
            
        emotion, confidence = await loop.run_in_executor(None, analyze_text_sync)
        
        # Get emoji and description using emotion_analyzer methods
        emotion_analyzer = get_emotion_analyzer()
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

@app.get("/api/health")
async def health():
    # Lightweight readiness probe
    try:
        status = {
            "status": "ok",
            "text_ready": True,
        }
        # Check audio status only if analyzer is already loaded
        status["audio_loaded"] = _audio_analyzer is not None
        if _audio_analyzer:
            try:
                audio_status = _audio_analyzer.get_runtime_status()
                status["audio_libs"] = bool(audio_status.get("hf_libs_available"))
            except Exception:
                status["audio_libs"] = False
        else:
            status["audio_libs"] = "not_loaded"
        return JSONResponse(content=status)
    except Exception as e:
        return JSONResponse(content={"status": "error", "error": str(e)}, status_code=500)

@app.get("/api/audio_status")
async def audio_status():
    try:
        analyzer = get_audio_analyzer()
        status = analyzer.get_runtime_status()
    except Exception as e:
        status = {"error": str(e)}
    return JSONResponse(content=status)

@app.post("/api/analyze_audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        if not os.path.exists("logs"):
            os.makedirs("logs")
        file_path = os.path.join("logs", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Offload CPU/GPU-bound audio analysis to a worker thread
        loop = asyncio.get_event_loop()
        
        def analyze_audio_sync():
            analyzer = get_audio_analyzer()
            return analyzer.analyze_audio(file_path)
            
        emotion, confidence = await loop.run_in_executor(None, analyze_audio_sync)
        
        # Get emoji and description
        emotion_analyzer = get_emotion_analyzer()
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
        
        def process_sync():
            # Heavy CPU work off the event loop
            import base64
            import numpy as np

            try:
                # Get cv2 lazily
                cv2 = get_cv2()
                
                # Remove data URL prefix if present
                img_data = image_data.split(",")[1] if image_data.startswith("data:image") else image_data
                image_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                
                # Validate array before decoding
                if len(nparr) == 0:
                    return {"error": "Empty image data", "status": 400}
                
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    return {"error": "Could not decode image - invalid format", "status": 400}
                    
            except Exception as decode_error:
                return {"error": f"Image decoding failed: {str(decode_error)}", "status": 400}

            try:
                # Get detectors lazily
                detector = get_face_detector()
                emotion_analyzer = get_emotion_analyzer()

                faces, _ = detector.detect_faces(image)
                if len(faces) == 0:
                    return {"faces": []}

                face_results = []
                for face_location in faces:
                    face_img = detector.extract_face(image, face_location)
                    if not detector.is_valid_face(face_img):
                        continue
                    emotion, confidence = emotion_analyzer.analyze_emotion(face_img)
                    emoji = emotion_analyzer.get_emotion_emoji(emotion)
                    description = emotion_analyzer.get_emotion_description(emotion, confidence)
                    face_results.append({
                        "emotion": emotion,
                        "confidence": confidence,
                        "emoji": emoji,
                        "description": description
                    })
                return {"faces": face_results}
            except Exception as analysis_error:
                return {"error": f"Face analysis failed: {str(analysis_error)}", "status": 500}

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, process_sync)
        if "error" in result:
            return JSONResponse(content={"error": result["error"]}, status_code=result.get("status", 400))
        return JSONResponse(content=result)
        
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
        # Prepare directories/paths
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        filename = f"emotion_{timestamp}{file_extension}"
        file_path = os.path.join("screenshots", filename)

        # Read file bytes once (async), then offload disk write to a thread
        data = await file.read()
        loop = asyncio.get_event_loop()
        def _write():
            with open(file_path, "wb") as f:
                f.write(data)
        await loop.run_in_executor(None, _write)

        return JSONResponse(content={"success": True, "filename": filename, "path": file_path})
        
    except Exception as e:
        logging.error(f"Screenshot save error: {str(e)}")
        return JSONResponse(
            content={"error": f"Failed to save screenshot: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
