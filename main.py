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
    try:
        # Reduce OpenCV thread usage to avoid startup CPU contention (optional)
        import os as _os
        threads_str = _os.environ.get("OPENCV_THREADS", "")
        if threads_str:
            try:
                cv2.setNumThreads(int(threads_str))
            except Exception:
                pass
    except Exception:
        pass

    if os.environ.get("PREWARM", "0") == "1":
        async def warm():
            try:
                # small text to trigger model load
                text_analyzer.analyze_text("hello")
            except Exception:
                pass
        asyncio.create_task(warm())
    # Optional: Pre-warm audio Whisper SER to cut first-run latency if PREWARM_AUDIO=1
    if os.environ.get("PREWARM_AUDIO", "0") == "1":
        async def warm_audio():
            try:
                # Lazy ensure model in background
                audio_analyzer._ensure_hf_model()
            except Exception:
                pass
        asyncio.create_task(warm_audio())
    # Optional: Pre-warm face detector to remove first-vision-call latency
    if os.environ.get("PREWARM_VISION", "0") == "1":
        async def warm_vision():
            try:
                global _face_detector_instance
                if _face_detector_instance is None:
                    _face_detector_instance = FaceDetector()
            except Exception:
                pass
        asyncio.create_task(warm_vision())

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
        emotion, confidence = await loop.run_in_executor(None, lambda: text_analyzer.analyze_text(text))
        
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

@app.get("/api/health")
async def health():
    # Lightweight readiness probe
    try:
        status = {
            "status": "ok",
            "text_ready": True,
        }
        # Optional check that won't block: has audio libs
        try:
            audio_status = audio_analyzer.get_runtime_status()
            status["audio_libs"] = bool(audio_status.get("hf_libs_available"))
            status["audio_loaded"] = bool(audio_status.get("hf_loaded"))
        except Exception:
            status["audio_libs"] = False
            status["audio_loaded"] = False
        return JSONResponse(content=status)
    except Exception as e:
        return JSONResponse(content={"status": "error", "error": str(e)}, status_code=500)

@app.get("/api/audio_status")
async def audio_status():
    try:
        # Attempt to pre-load HF model (lazy)
        _ = audio_analyzer._ensure_hf_model()
    except Exception:
        pass
    try:
        status = audio_analyzer.get_runtime_status()
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
        emotion, confidence = await loop.run_in_executor(None, lambda: audio_analyzer.analyze_audio(file_path))
        
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
        
        def process_sync():
            # Heavy CPU work off the event loop
            import base64
            import numpy as np

            # Remove data URL prefix if present
            img_data = image_data.split(",")[1] if image_data.startswith("data:image") else image_data
            image_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return {"error": "Could not decode image", "status": 400}

            # Initialize on first use
            global _face_detector_instance
            if _face_detector_instance is None:
                _face_detector_instance = FaceDetector()
            detector = _face_detector_instance

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
