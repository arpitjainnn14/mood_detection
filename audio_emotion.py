import logging
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

try:
    import librosa
    import soundfile as sf
except Exception:
    librosa = None
    sf = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
except Exception:
    RandomForestClassifier = None
    StandardScaler = None
    RobustScaler = None
    Pipeline = None
    cross_val_score = None

logger = logging.getLogger(__name__)

class AudioEmotionAnalyzer:
    """
    Advanced Audio Emotion Analyzer using comprehensive feature extraction
    and sophisticated classification methods for better emotion recognition.
    """
    
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_mapping = {
            'angry': {'valence': -0.7, 'arousal': 0.8, 'dominance': 0.6},
            'disgust': {'valence': -0.8, 'arousal': 0.3, 'dominance': -0.2},
            'fear': {'valence': -0.6, 'arousal': 0.7, 'dominance': -0.8},
            'happy': {'valence': 0.8, 'arousal': 0.6, 'dominance': 0.5},
            'sad': {'valence': -0.8, 'arousal': -0.5, 'dominance': -0.6},
            'surprise': {'valence': 0.2, 'arousal': 0.9, 'dominance': 0.1},
            'neutral': {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}
        }
        
        # Initialize advanced classifier if available
        self.classifier = None
        if RandomForestClassifier and StandardScaler:
            self.classifier = Pipeline([
                ('scaler', RobustScaler() if RobustScaler else StandardScaler()),
                ('rf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])
            # Pre-train with synthetic data for better baseline
            self._initialize_classifier()

    def _initialize_classifier(self):
        """Initialize classifier with synthetic training data based on emotion characteristics."""
        if self.classifier is None:
            return
        
        # Generate synthetic training data based on emotion characteristics
        np.random.seed(42)
        n_samples_per_emotion = 50
        feature_dim = 120  # Expected feature dimension
        
        X_synthetic = []
        y_synthetic = []
        
        for emotion in self.emotions:
            props = self.emotion_mapping[emotion]
            
            for _ in range(n_samples_per_emotion):
                # Generate features based on emotion characteristics
                features = self._generate_emotion_features(props, feature_dim)
                X_synthetic.append(features)
                y_synthetic.append(emotion)
        
        X_synthetic = np.array(X_synthetic)
        y_synthetic = np.array(y_synthetic)
        
        try:
            self.classifier.fit(X_synthetic, y_synthetic)
            logger.info("Audio emotion classifier initialized with synthetic data")
        except Exception as e:
            logger.warning(f"Failed to initialize classifier: {e}")
            self.classifier = None
    
    def _generate_emotion_features(self, emotion_props: Dict, feature_dim: int) -> np.ndarray:
        """Generate synthetic features based on emotion properties."""
        valence = emotion_props['valence']
        arousal = emotion_props['arousal']
        dominance = emotion_props['dominance']
        
        # Base features with emotion-specific characteristics
        features = np.random.normal(0, 1, feature_dim)
        
        # Modulate features based on emotion properties
        features[:20] += valence * 2  # MFCC influenced by valence
        features[20:40] += arousal * 1.5  # Spectral features influenced by arousal
        features[40:60] += dominance * 1.2  # Rhythm features influenced by dominance
        features[60:80] += (valence + arousal) * 0.8  # Prosodic features
        
        # Add some noise and normalize
        features += np.random.normal(0, 0.3, feature_dim)
        return features

    def extract_comprehensive_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract comprehensive audio features for emotion recognition."""
        if librosa is None:
            return np.zeros((120,), dtype=float)
        
        try:
            # Preprocessing
            y = self._preprocess_audio(y, sr)
            
            # 1. MFCC Features (20 features)
            mfcc_features = self._extract_mfcc_features(y, sr)
            
            # 2. Spectral Features (20 features)
            spectral_features = self._extract_spectral_features(y, sr)
            
            # 3. Rhythm and Temporal Features (20 features)
            rhythm_features = self._extract_rhythm_features(y, sr)
            
            # 4. Prosodic Features (20 features)
            prosodic_features = self._extract_prosodic_features(y, sr)
            
            # 5. Harmonic and Timbral Features (20 features)
            harmonic_features = self._extract_harmonic_features(y, sr)
            
            # 6. Statistical Features (20 features)
            statistical_features = self._extract_statistical_features(y, sr)
            
            # Combine all features
            all_features = np.concatenate([
                mfcc_features, spectral_features, rhythm_features,
                prosodic_features, harmonic_features, statistical_features
            ])
            
            # Handle NaN/Inf values
            all_features = np.nan_to_num(all_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return all_features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros((120,), dtype=float)
    
    def _preprocess_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Advanced audio preprocessing for better feature extraction."""
        try:
            # 1. Normalize audio to prevent clipping
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # 2. Apply high-pass filter to remove DC offset and low-frequency noise
            y = self._apply_highpass_filter(y, sr, cutoff=80)
            
            # 3. Remove silence and padding
            y_trimmed, _ = librosa.effects.trim(y, top_db=20, frame_length=2048, hop_length=512)
            if len(y_trimmed) == 0:
                return y
            
            # 4. Apply pre-emphasis filter to balance frequency spectrum
            y_emphasized = np.append(y_trimmed[0], y_trimmed[1:] - 0.97 * y_trimmed[:-1])
            
            # 5. Apply soft noise reduction using spectral gating
            y_denoised = self._spectral_noise_reduction(y_emphasized, sr)
            
            # 6. Ensure minimum length for stable feature extraction
            min_length = int(0.5 * sr)  # 0.5 seconds minimum
            if len(y_denoised) < min_length:
                # Pad with reflection to avoid artifacts
                y_denoised = np.pad(y_denoised, (0, min_length - len(y_denoised)), mode='reflect')
            
            # 7. Final normalization
            if np.max(np.abs(y_denoised)) > 0:
                y_denoised = y_denoised / np.max(np.abs(y_denoised)) * 0.95
            
            return y_denoised
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}, using basic preprocessing")
            # Fallback to basic preprocessing
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            return y_trimmed if len(y_trimmed) > 0 else y
    
    def _apply_highpass_filter(self, y: np.ndarray, sr: int, cutoff: float = 80) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise."""
        try:
            from scipy.signal import butter, filtfilt
            # Design butterworth high-pass filter
            nyquist = sr / 2
            normalized_cutoff = cutoff / nyquist
            b, a = butter(5, normalized_cutoff, btype='high', analog=False)
            # Apply filter
            return filtfilt(b, a, y)
        except:
            # Fallback: simple high-pass using differencing
            return np.diff(y, prepend=y[0])
    
    def _spectral_noise_reduction(self, y: np.ndarray, sr: int, noise_floor_db: float = -40) -> np.ndarray:
        """Apply spectral noise reduction using adaptive noise floor estimation."""
        try:
            # Compute STFT
            stft = librosa.stft(y, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor from quiet segments
            frame_energy = np.mean(magnitude, axis=0)
            noise_threshold = np.percentile(frame_energy, 20)  # Bottom 20% as noise estimate
            
            # Apply spectral subtraction
            noise_mask = frame_energy <= noise_threshold
            if np.any(noise_mask):
                noise_spectrum = np.mean(magnitude[:, noise_mask], axis=1, keepdims=True)
                
                # Spectral subtraction with over-subtraction factor
                alpha = 2.0  # Over-subtraction factor
                magnitude_clean = magnitude - alpha * noise_spectrum
                
                # Apply spectral floor to prevent over-subtraction artifacts
                spectral_floor = 0.1 * magnitude
                magnitude_clean = np.maximum(magnitude_clean, spectral_floor)
            else:
                magnitude_clean = magnitude
            
            # Reconstruct audio
            stft_clean = magnitude_clean * np.exp(1j * phase)
            y_clean = librosa.istft(stft_clean, hop_length=512)
            
            return y_clean
            
        except Exception as e:
            logger.warning(f"Spectral noise reduction failed: {e}")
            return y
    
    def _extract_mfcc_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features with deltas."""
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # Statistical measures
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        return np.concatenate([mfcc_mean[:7], mfcc_std[:7], np.mean(mfcc_delta, axis=1)[:6]])
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectral features."""
        # Spectral centroid, bandwidth, rolloff
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        features = [
            np.mean(spectral_centroids), np.std(spectral_centroids),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(zcr), np.std(zcr),
            np.mean(rms), np.std(rms)
        ]
        
        # Add spectral contrast statistics
        contrast_stats = [np.mean(spectral_contrast), np.std(spectral_contrast)]
        for i in range(min(8, spectral_contrast.shape[0])):
            contrast_stats.extend([np.mean(spectral_contrast[i]), np.std(spectral_contrast[i])])
        
        features.extend(contrast_stats[:10])
        return np.array(features[:20])
    
    def _extract_rhythm_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract rhythm and temporal features."""
        features = []
        
        # Tempo and beat tracking
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features.extend([tempo, len(beats) / (len(y) / sr)])  # tempo, beat density
            
            # Beat intervals
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                features.extend([
                    np.mean(beat_intervals), np.std(beat_intervals),
                    np.min(beat_intervals), np.max(beat_intervals)
                ])
            else:
                features.extend([0, 0, 0, 0])
        except:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Onset detection
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            # Onset statistics
            features.extend([
                len(onset_times) / (len(y) / sr),  # onset rate
                np.mean(np.diff(onset_times)) if len(onset_times) > 1 else 0,  # mean onset interval
                np.std(np.diff(onset_times)) if len(onset_times) > 1 else 0   # onset interval variance
            ])
        except:
            features.extend([0, 0, 0])
        
        # Pad to 20 features
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])
    
    def _extract_prosodic_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract prosodic features related to pitch and intonation."""
        features = []
        
        # Fundamental frequency (F0) estimation
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
            )
            
            # Remove unvoiced frames
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 0:
                features.extend([
                    np.mean(f0_voiced), np.std(f0_voiced),
                    np.min(f0_voiced), np.max(f0_voiced),
                    np.median(f0_voiced), stats.iqr(f0_voiced),
                    np.sum(voiced_flag) / len(voiced_flag)  # voicing ratio
                ])
                
                # Pitch contour features
                if len(f0_voiced) > 1:
                    f0_diff = np.diff(f0_voiced)
                    features.extend([
                        np.mean(f0_diff), np.std(f0_diff),
                        np.sum(f0_diff > 0) / len(f0_diff),  # rising ratio
                        np.sum(f0_diff < 0) / len(f0_diff)   # falling ratio
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            else:
                features.extend([0] * 11)
        except:
            features.extend([0] * 11)
        
        # Jitter and shimmer (voice quality measures)
        try:
            # Simple jitter approximation
            if len(f0_voiced) > 1:
                jitter = np.mean(np.abs(np.diff(f0_voiced)) / f0_voiced[:-1])
                features.append(jitter)
            else:
                features.append(0)
        except:
            features.append(0)
        
        # Pad to 20 features
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])
    
    def _extract_harmonic_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract harmonic and timbral features."""
        features = []
        
        # Harmonic-percussive separation
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Harmonic ratio
            harmonic_energy = np.sum(y_harmonic ** 2)
            percussive_energy = np.sum(y_percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            if total_energy > 0:
                harmonic_ratio = harmonic_energy / total_energy
            else:
                harmonic_ratio = 0
            
            features.append(harmonic_ratio)
            
            # Harmonic spectral features
            harmonic_centroids = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)[0]
            features.extend([np.mean(harmonic_centroids), np.std(harmonic_centroids)])
            
        except:
            features.extend([0, 0, 0])
        
        # Chroma features
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_stats = [np.mean(chroma), np.std(chroma)]
            
            # Add individual chroma means (12 pitch classes, take first 10)
            for i in range(min(10, chroma.shape[0])):
                chroma_stats.append(np.mean(chroma[i]))
            
            features.extend(chroma_stats[:12])
        except:
            features.extend([0] * 12)
        
        # Tonnetz (tonal centroid features)
        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features.extend([np.mean(tonnetz), np.std(tonnetz)])
        except:
            features.extend([0, 0])
        
        # Pad to 20 features
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])
    
    def _extract_statistical_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract statistical features from the audio signal."""
        features = []
        
        # Time domain statistics
        features.extend([
            np.mean(y), np.std(y), stats.skew(y), stats.kurtosis(y),
            np.min(y), np.max(y), np.median(y), stats.iqr(y)
        ])
        
        # Frequency domain statistics
        try:
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # Spectral statistics
            features.extend([
                np.mean(magnitude), np.std(magnitude),
                stats.skew(magnitude.flatten()), stats.kurtosis(magnitude.flatten())
            ])
            
            # Spectral flux (measure of spectral change)
            spectral_flux = np.mean(np.diff(magnitude, axis=1) ** 2)
            features.append(spectral_flux)
            
        except:
            features.extend([0, 0, 0, 0, 0])
        
        # Energy and power features
        features.extend([
            np.sum(y ** 2),  # total energy
            np.mean(y ** 2),  # average power
            np.std(y ** 2)   # power variance
        ])
        
        # Zero-crossing rate variance
        try:
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([np.mean(zcr), np.std(zcr)])
        except:
            features.extend([0, 0])
        
        return np.array(features[:20])

    def analyze_audio(self, audio_path: str) -> Tuple[str, float]:
        """
        Analyze audio file for emotion using comprehensive feature extraction
        and advanced classification methods.
        """
        if not audio_path:
            return 'neutral', 0.0
            
        if librosa is None:
            logger.warning("librosa not available. Install 'librosa' to enable audio emotion.")
            return 'neutral', 0.0
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=22050, mono=True)  # Standardize sample rate
            if y is None or len(y) == 0:
                return 'neutral', 0.0
            
            # Extract comprehensive features
            features = self.extract_comprehensive_features(y, sr)
            if features.size == 0:
                return 'neutral', 0.0
            
            # Use advanced classifier if available
            if self.classifier is not None:
                try:
                    # Predict using trained classifier
                    prediction = self.classifier.predict([features])[0]
                    probabilities = self.classifier.predict_proba([features])[0]
                    
                    # Get confidence for the predicted emotion
                    emotion_idx = list(self.classifier.classes_).index(prediction)
                    confidence = float(probabilities[emotion_idx])
                    
                    return prediction, confidence
                    
                except Exception as e:
                    logger.warning(f"Classifier prediction failed: {e}, falling back to ensemble method")
            
            # Fallback to ensemble method with multiple approaches
            return self._ensemble_emotion_prediction(features, y, sr)
            
        except Exception as e:
            logger.error(f"Audio emotion analysis failed: {e}")
            return 'neutral', 0.0
    
    def _ensemble_emotion_prediction(self, features: np.ndarray, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """
        Ensemble method combining multiple emotion prediction approaches.
        """
        predictions = {}
        
        # Method 1: Valence-Arousal-Dominance (VAD) based prediction
        vad_prediction = self._predict_from_vad(features, y, sr)
        predictions['vad'] = vad_prediction
        
        # Method 2: Spectral pattern analysis
        spectral_prediction = self._predict_from_spectral_patterns(features, y, sr)
        predictions['spectral'] = spectral_prediction
        
        # Method 3: Prosodic pattern analysis
        prosodic_prediction = self._predict_from_prosodic_patterns(features, y, sr)
        predictions['prosodic'] = prosodic_prediction
        
        # Method 4: Energy and rhythm analysis
        energy_prediction = self._predict_from_energy_rhythm(features, y, sr)
        predictions['energy'] = energy_prediction
        
        # Combine predictions using weighted voting
        final_prediction = self._combine_predictions(predictions)
        return final_prediction
    
    def _predict_from_vad(self, features: np.ndarray, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """Predict emotion based on Valence-Arousal-Dominance model."""
        try:
            # Extract key features for VAD estimation
            rms = np.mean(librosa.feature.rms(y=y)[0])
            zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            
            # Estimate VAD values
            arousal = min(1.0, (rms * 10 + zcr * 5 + spectral_centroid / 2000) / 3)
            valence = min(1.0, max(-1.0, (spectral_centroid / 2000 - 0.5) * 2))
            dominance = min(1.0, max(-1.0, (rms * 5 - 0.3) * 2))
            
            # Find closest emotion in VAD space
            min_distance = float('inf')
            best_emotion = 'neutral'
            
            for emotion, vad_values in self.emotion_mapping.items():
                distance = np.sqrt(
                    (valence - vad_values['valence']) ** 2 +
                    (arousal - vad_values['arousal']) ** 2 +
                    (dominance - vad_values['dominance']) ** 2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    best_emotion = emotion
            
            # Convert distance to confidence (closer = higher confidence)
            confidence = max(0.1, 1.0 - min_distance / 2.0)
            return best_emotion, confidence
            
        except Exception as e:
            logger.warning(f"VAD prediction failed: {e}")
            return 'neutral', 0.1
    
    def _predict_from_spectral_patterns(self, features: np.ndarray, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """Predict emotion based on spectral patterns."""
        try:
            # Extract spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
            
            # Normalize features
            centroid_norm = spectral_centroid / 4000  # Normalize to 0-1 range approximately
            bandwidth_norm = spectral_bandwidth / 2000
            rolloff_norm = spectral_rolloff / 8000
            
            # Emotion-specific spectral patterns
            patterns = {
                'angry': (0.7, 0.8, 0.8),    # High centroid, high bandwidth, high rolloff
                'happy': (0.6, 0.6, 0.7),    # Medium-high values
                'surprise': (0.8, 0.9, 0.9), # Very high values
                'fear': (0.5, 0.7, 0.6),     # Medium centroid, high bandwidth
                'sad': (0.3, 0.4, 0.4),      # Low values
                'disgust': (0.4, 0.5, 0.5),  # Low-medium values
                'neutral': (0.5, 0.5, 0.5)   # Medium values
            }
            
            # Find best match
            min_distance = float('inf')
            best_emotion = 'neutral'
            
            for emotion, (exp_cent, exp_band, exp_roll) in patterns.items():
                distance = np.sqrt(
                    (centroid_norm - exp_cent) ** 2 +
                    (bandwidth_norm - exp_band) ** 2 +
                    (rolloff_norm - exp_roll) ** 2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    best_emotion = emotion
            
            confidence = max(0.1, 1.0 - min_distance)
            return best_emotion, confidence
            
        except Exception as e:
            logger.warning(f"Spectral pattern prediction failed: {e}")
            return 'neutral', 0.1
    
    def _predict_from_prosodic_patterns(self, features: np.ndarray, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """Predict emotion based on prosodic patterns (pitch, rhythm)."""
        try:
            # Extract pitch information
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
            )
            
            if np.sum(voiced_flag) == 0:
                return 'neutral', 0.1
            
            f0_voiced = f0[voiced_flag]
            
            # Prosodic features
            pitch_mean = np.mean(f0_voiced)
            pitch_std = np.std(f0_voiced)
            pitch_range = np.max(f0_voiced) - np.min(f0_voiced)
            voicing_ratio = np.sum(voiced_flag) / len(voiced_flag)
            
            # Emotion-specific prosodic patterns
            # Based on research in speech emotion recognition
            if pitch_mean > 200 and pitch_std > 30:  # High pitch, high variation
                if pitch_range > 100:
                    return 'surprise', 0.8
                else:
                    return 'happy', 0.7
            elif pitch_mean > 180 and pitch_std > 25:  # Medium-high pitch, medium variation
                return 'angry', 0.7
            elif pitch_mean < 150 and pitch_std < 20:  # Low pitch, low variation
                if voicing_ratio < 0.6:
                    return 'sad', 0.7
                else:
                    return 'disgust', 0.6
            elif pitch_std > 35:  # High variation regardless of mean
                return 'fear', 0.6
            else:
                return 'neutral', 0.5
                
        except Exception as e:
            logger.warning(f"Prosodic pattern prediction failed: {e}")
            return 'neutral', 0.1
    
    def _predict_from_energy_rhythm(self, features: np.ndarray, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """Predict emotion based on energy and rhythm patterns."""
        try:
            # Energy features
            rms = np.mean(librosa.feature.rms(y=y)[0])
            energy_std = np.std(librosa.feature.rms(y=y)[0])
            
            # Rhythm features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Energy-based emotion patterns
            if rms > 0.1:  # High energy
                if energy_std > 0.05:  # High energy variation
                    if tempo > 120:
                        return 'angry', 0.8
                    else:
                        return 'surprise', 0.7
                else:  # Stable high energy
                    return 'happy', 0.7
            elif rms < 0.05:  # Low energy
                if tempo < 80:
                    return 'sad', 0.8
                else:
                    return 'fear', 0.6
            else:  # Medium energy
                if energy_std > 0.03:
                    return 'disgust', 0.6
                else:
                    return 'neutral', 0.6
                    
        except Exception as e:
            logger.warning(f"Energy-rhythm prediction failed: {e}")
            return 'neutral', 0.1
    
    def _combine_predictions(self, predictions: Dict[str, Tuple[str, float]]) -> Tuple[str, float]:
        """Combine multiple predictions using weighted voting."""
        # Weights for different methods
        weights = {
            'vad': 0.3,
            'spectral': 0.25,
            'prosodic': 0.25,
            'energy': 0.2
        }
        
        # Collect weighted votes
        emotion_scores = {emotion: 0.0 for emotion in self.emotions}
        total_weight = 0.0
        
        for method, (emotion, confidence) in predictions.items():
            if method in weights:
                weight = weights[method] * confidence
                emotion_scores[emotion] += weight
                total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            emotion_scores = {k: v / total_weight for k, v in emotion_scores.items()}
        
        # Find best emotion
        best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        # Calculate overall confidence
        confidence = min(0.95, max(0.1, best_emotion[1]))
        
        return best_emotion[0], confidence
    
    def get_detailed_analysis(self, audio_path: str) -> Dict:
        """
        Get detailed analysis results for debugging and transparency.
        """
        if not audio_path or librosa is None:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'error': 'Audio path not provided or librosa not available'
            }
        
        try:
            # Load and preprocess audio
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            if y is None or len(y) == 0:
                return {'emotion': 'neutral', 'confidence': 0.0, 'error': 'Empty audio file'}
            
            # Extract features
            features = self.extract_comprehensive_features(y, sr)
            
            # Get individual predictions
            vad_pred = self._predict_from_vad(features, y, sr)
            spectral_pred = self._predict_from_spectral_patterns(features, y, sr)
            prosodic_pred = self._predict_from_prosodic_patterns(features, y, sr)
            energy_pred = self._predict_from_energy_rhythm(features, y, sr)
            
            # Get final prediction
            predictions = {
                'vad': vad_pred,
                'spectral': spectral_pred,
                'prosodic': prosodic_pred,
                'energy': energy_pred
            }
            final_pred = self._combine_predictions(predictions)
            
            # Extract basic audio characteristics
            duration = len(y) / sr
            rms_energy = np.mean(librosa.feature.rms(y=y)[0])
            zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
            
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            except:
                tempo = 0
                spectral_centroid = 0
            
            return {
                'emotion': final_pred[0],
                'confidence': final_pred[1],
                'individual_predictions': {
                    'vad_model': {'emotion': vad_pred[0], 'confidence': vad_pred[1]},
                    'spectral_model': {'emotion': spectral_pred[0], 'confidence': spectral_pred[1]},
                    'prosodic_model': {'emotion': prosodic_pred[0], 'confidence': prosodic_pred[1]},
                    'energy_model': {'emotion': energy_pred[0], 'confidence': energy_pred[1]}
                },
                'audio_characteristics': {
                    'duration_seconds': round(duration, 2),
                    'sample_rate': sr,
                    'rms_energy': round(float(rms_energy), 4),
                    'zero_crossing_rate': round(float(zcr), 4),
                    'tempo_bpm': round(float(tempo), 1),
                    'spectral_centroid_hz': round(float(spectral_centroid), 1)
                },
                'feature_vector_size': len(features),
                'preprocessing_applied': True
            }
            
        except Exception as e:
            logger.error(f"Detailed analysis failed: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
