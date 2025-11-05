from deepface import DeepFace
import numpy as np
import cv2
import logging
from collections import deque
from typing import Tuple, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    def __init__(self, settings: Optional[object] = None):
        self.settings = settings  # optional Settings instance (from settings.py)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        # Configurable smoothing window (defaults to 3)
        smoothing = 3
        try:
            if self.settings is not None:
                smoothing = max(1, int(self.settings.get('emotion_smoothing', 3)))
        except Exception:
            smoothing = 3

        self.emotion_history = deque(maxlen=smoothing)
        self.confidence_history = deque(maxlen=smoothing)

        # Neutral often dominates. Keep weights mostly equal; slightly downweight neutral.
        self.emotion_weights: Dict[str, float] = {
            'happy': 1.0,
            'surprise': 1.0,
            'angry': 1.0,
            'fear': 1.0,
            'sad': 1.0,
            'disgust': 1.0,
            'neutral': 0.9
        }

        # Confidence thresholds; raise neutral to reduce bias, allow others easier.
        self.confidence_thresholds: Dict[str, float] = {
            'happy': 0.20,
            'surprise': 0.20,
            'neutral': 0.50,
            'default': 0.20
        }

        # If user chooses quality, we can be stricter on thresholds for robustness.
        try:
            dq = (self.settings.get('detection_quality') if self.settings else 'balanced') or 'balanced'
            if dq == 'quality':
                self.confidence_thresholds['default'] = 0.25
                self.confidence_thresholds['neutral'] = 0.55
        except Exception:
            pass

        # Use a margin between top-1 and top-2 to avoid neutral/happy dominance when close.
        self.top2_margin: float = 0.15

    def _select_backend(self) -> Tuple[str, bool, bool]:
        """Select DeepFace backend and options based on settings.

        Returns:
            (backend, align, enforce_detection)
        """
        backend = 'opencv'
        align = False
        enforce = False
        try:
            quality = (self.settings.get('detection_quality') if self.settings else 'balanced') or 'balanced'
        except Exception:
            quality = 'balanced'

        if quality == 'performance':
            backend, align, enforce = 'opencv', False, False
        elif quality == 'quality':
            # Best accuracy; slower
            backend, align, enforce = 'retinaface', True, True
        else:
            # balanced
            backend, align, enforce = 'mediapipe', True, True

        return backend, align, enforce

    def enhance_contrast(self, img):
        """Fast contrast enhancement for real-time processing."""
        if len(img.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Quick CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge and convert back
            return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
        return img

    def preprocess_face(self, face_img):
        """Fast face preprocessing for real-time detection."""
        # Quick resize if needed
        if face_img.shape[0] < 96 or face_img.shape[1] < 96:
            face_img = cv2.resize(face_img, (96, 96))
        
        # Basic contrast enhancement
        face_img = self.enhance_contrast(face_img)
        
        return face_img

    def analyze_emotion(self, face_img) -> Tuple[str, float]:
        """
        Fast emotion analysis optimized for real-time performance.
        
        Args:
            face_img: Face image (BGR format)
            
        Returns:
            tuple: (dominant_emotion, confidence)
        """
        try:
            # Basic validation
            if face_img is None or face_img.size == 0:
                return 'neutral', 0.0

            # Quick preprocessing
            processed_face = self.preprocess_face(face_img)
            
            # Select backend/alignment based on quality setting
            backend, align, enforce = self._select_backend()

            # Single analysis
            try:
                result = DeepFace.analyze(
                    processed_face,
                    actions=['emotion'],
                    enforce_detection=enforce,
                    align=align,
                    detector_backend=backend,
                    silent=True
                )
            except Exception as e:
                logger.debug(f"Primary analysis failed (backend={backend}): {str(e)}; retrying with OpenCV fallback")
                try:
                    result = DeepFace.analyze(
                        processed_face,
                        actions=['emotion'],
                        enforce_detection=False,
                        align=False,
                        detector_backend='opencv',
                        silent=True
                    )
                except Exception as e2:
                    logger.debug(f"Fallback analysis failed: {str(e2)}")
                    return 'neutral', 0.0

            # DeepFace may return a list or a dict depending on version
            try:
                analysis = result[0] if isinstance(result, (list, tuple)) else result
                emotions = analysis.get('emotion', {})
            except Exception as e:
                logger.debug(f"Unexpected analysis result format: {type(result)} - {str(e)}")
                return 'neutral', 0.0

            # Apply emotion weights
            weighted_emotions = {
                emo: emotions[emo] * self.emotion_weights.get(emo, 1.0)
                for emo in self.emotions
            }
            
            # Sort emotions by weighted score (descending)
            sorted_weighted = sorted(weighted_emotions.items(), key=lambda x: x[1], reverse=True)
            top1_emotion, _ = sorted_weighted[0]
            top2_emotion, _ = sorted_weighted[1]

            # Use raw probabilities for confidence and margin checks
            top1_conf = emotions[top1_emotion] / 100.0
            top2_conf = emotions[top2_emotion] / 100.0
            
            # Check confidence threshold
            threshold_top1 = self.confidence_thresholds.get(top1_emotion, self.confidence_thresholds['default'])

            chosen_emotion = top1_emotion
            chosen_conf = top1_conf

            # If top-1 is below its threshold, try to promote a better alternative that clears its threshold.
            if top1_conf < threshold_top1:
                for emotion, _ in sorted_weighted[1:]:
                    conf = emotions[emotion] / 100.0
                    if conf >= self.confidence_thresholds.get(emotion, self.confidence_thresholds['default']):
                        chosen_emotion = emotion
                        chosen_conf = conf
                        break

            # Reduce neutral/happy dominance: if top2 is close and is a non-neutral alternative, prefer it.
            if chosen_emotion in ('neutral', 'happy'):
                if (top1_conf - top2_conf) < self.top2_margin and top2_emotion != chosen_emotion:
                    # Ensure top-2 clears its threshold
                    if top2_conf >= self.confidence_thresholds.get(top2_emotion, self.confidence_thresholds['default']):
                        chosen_emotion = top2_emotion
                        chosen_conf = top2_conf
            
            # Update history
            self.emotion_history.append(chosen_emotion)
            self.confidence_history.append(chosen_conf)
            
            # Simple temporal smoothing
            if len(self.emotion_history) >= 2:
                if self.emotion_history[-1] == self.emotion_history[-2]:
                    return chosen_emotion, chosen_conf
            
            return chosen_emotion, chosen_conf
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            return 'neutral', 0.1
    
    def get_emotion_color(self, emotion):
        """
        Get color for emotion visualization.
        
        Args:
            emotion: Emotion name
            
        Returns:
            tuple: BGR color values
        """
        color_map = {
            'happy': (0, 255, 255),     # Yellow
            'sad': (255, 128, 0),       # Orange
            'angry': (0, 0, 255),       # Red
            'surprise': (255, 255, 0),   # Cyan
            'fear': (255, 0, 255),      # Magenta
            'disgust': (0, 255, 0),     # Green
            'neutral': (255, 255, 255),  # White
            'unknown': (128, 128, 128)   # Gray
        }
        return color_map.get(emotion, (128, 128, 128))
    
    def get_emotion_emoji(self, emotion):
        """
        Get emoji for emotion visualization.
        
        Args:
            emotion: Emotion name
            
        Returns:
            str: Emoji character
        """
        emoji_map = {
            'happy': '😄',
            'sad': '😢',
            'angry': '😡',
            'surprise': '😮',
            'fear': '😨',
            'disgust': '🤢',
            'neutral': '😐',
            'unknown': '❓'
        }
        return emoji_map.get(emotion, '❓')
    
    def get_emotion_description(self, emotion, confidence):
        """
        Get a description of the emotion.
        
        Args:
            emotion: Emotion name
            confidence: Confidence score
            
        Returns:
            str: Description of the emotion
        """
        descriptions = {
            'happy': f"Happy ({confidence:.0%} confidence)",
            'sad': f"Sad ({confidence:.0%} confidence)",
            'angry': f"Angry ({confidence:.0%} confidence)",
            'surprise': f"Surprised ({confidence:.0%} confidence)",
            'fear': f"Afraid ({confidence:.0%} confidence)",
            'disgust': f"Disgusted ({confidence:.0%} confidence)",
            'neutral': f"Neutral ({confidence:.0%} confidence)",
            'unknown': "Unable to determine emotion"
        }
        return descriptions.get(emotion, "Unable to determine emotion") 