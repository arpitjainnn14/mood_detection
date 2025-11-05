
# TheraVox: Multimodal Emotion Detection (Web)

TheraVox is a FastAPI web app that detects emotions from three modalities:
- Vision (webcam snapshot → face emotions via DeepFace)
- Text (Transformer model with rule‑based fallback)
- Audio (upload/record → analysis), including live microphone recording in the browser

The UI is responsive, dark-themed, and includes an animated welcome page plus dedicated pages for each modality.

## Features

- Vision: face detection and emotion classification with DeepFace backends.
- Text: Transformer‑based emotion (j‑hartmann/distilroberta) with lexicon fallback.
- Audio: upload (WAV/MP3) or record live; get emotion + confidence.
- Developers page for capstone credits.
- Clean, modern layout with subtle CSS animations.
- Wellness Toolkit: breathing coach, grounding prompts, 3‑minute reset, affirmations.

Detected emotions (7): Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral


## Tech Stack

- Python 3.8+
- FastAPI (routing + API)
- Uvicorn (ASGI server)
- OpenCV + DeepFace (vision)
- Transformers + PyTorch (text)
- librosa (audio)
- HTML/CSS/JS (no frontend framework)


## Setup (Windows, PowerShell)

1) Create and activate a virtual environment

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

3) Run the FastAPI server

```
uvicorn main:app --reload
```

Or simply run:

```
python main.py
```

4) Open in your browser
- Home (animated): http://localhost:8000/
- Vision: http://localhost:8000/vision
- Text: http://localhost:8000/text
- Audio (upload or live record): http://localhost:8000/audio
- Wellness Toolkit: http://localhost:8000/wellness
- Developers: http://localhost:8000/developers

Notes
- On the first DeepFace and Transformer calls, models may download; allow a minute on first run.
- Your browser will prompt for camera/microphone permissions.

## Logo

Place your project logo at:
- Preferred: `static/img/logo.svg`
- Fallback: `static/img/logo.png`

The header and hero will automatically use the SVG if available, otherwise PNG.

## Project Structure

- `main.py` — FastAPI app with page routes and REST APIs (`/api/analyze_frame`, `/api/analyze_text`, `/api/analyze_audio`)
  - `base.html` — shared layout (header/nav/footer)
  - `home.html` — animated welcome/landing page
  - `vision.html`, `text.html`, `audio.html`, `developers.html`, `wellness.html` — dedicated pages
  - `index.html` — combined analyzer page (optional)
  - `styles.css` — theme + animations
  - `app.js` — webcam capture, text/audio requests, live mic recording (WAV encode + upload)
  - `img/` — project images (logo.svg/png)

## Live Audio Recording

- Click Record, speak (auto‑stops around 15s), then Stop; the app encodes WAV client‑side and uploads to `/analyze_audio`.
- Requests auto‑timeout to avoid hanging; errors are shown inline.

## Troubleshooting

- If video/audio permissions are blocked, reload and allow access.
- Large/long audio may take longer to analyze; try 3–10s for quick checks.
- If `/analyze_audio` fails, check terminal logs for Python errors (librosa/soundfile backends).

## Model Details (Text)

- Primary model: `j-hartmann/emotion-english-distilroberta-base` via Hugging Face Transformers.
- Fallback: rule‑based analyzer with lexicons, negation, and intensifiers.
- Mapping: joy→happy, anger, sadness, fear, disgust, surprise, love→happy, neutral→neutral.

## License

MIT — see `LICENSE`.

## Credits

Capstone Developers: Arpit Jain, Bhoomi Chowksey, Ajit Dixit, Suchita Nandi, Aditi Bathla

Libraries: OpenCV, DeepFace, librosa