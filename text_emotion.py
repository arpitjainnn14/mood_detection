import re
from typing import Dict, List, Tuple, Optional

try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None  # Fallback if transformers isn't available

class TextEmotionAnalyzer:
    """
    Lighter, fixed version of your original analyzer with:
    - normalized text handling
    - set-based lexicons
    - safe exclamation boost (only if there is evidence)
    - small improvements to negation/intensifier handling
    """

    def __init__(self) -> None:
        # Lazy-initialized transformer pipeline
        self._hf_pipe = None  # type: Optional[object]
        # Core lexicons (use sets for fast membership checks).
        # Add common word-forms or consider lemmatization for completeness.
        self.lex: Dict[str, set] = {
            'happy': set(['happy','joy','joyful','glad','excited','love','lovely','awesome','great','amazing',
                          'fantastic','wonderful','delighted','pleased','smile','smiling','cheerful','proud',
                          'relieved','celebrate','congrats','congratulations','win','won','victory','yay',
                          'good','job','well','done','love','loved']),
            'sad': set(['sad','unhappy','down','depressed','cry','crying','tears','heartbroken','miserable',
                        'sorrow','grief','lonely','alone','hurt','disappointed','let','down','regret','loss',
                        'lost','pain','gloomy']),
            'angry': set(['angry','furious','annoyed','irritated','rage','mad','livid','outraged','pissed',
                          'resentful','hate','hating','frustrated','offended','insulted','bitter']),
            'fear': set(['afraid','scared','terrified','anxious','anxiety','worried','worry','panic','panicking',
                         'nervous','fear','fearful','frightened','concerned','unsafe','threat','danger','alarmed']),
            'disgust': set(['disgust','disgusted','disgusting','gross','nasty','revolting','repulsed','sickened',
                            'vomit','yuck','ew','filthy','nauseated','abhorrent','vile']),
            'surprise': set(['surprise','surprised','surprising','astonished','wow','shocked','unbelievable',
                             'unexpected','suddenly','amazed','what','no','way','omg']),
        }

        # Emoticons / emoji -> emotion
        self.emoji_map: Dict[str, str] = {
            ':)': 'happy', ':-)': 'happy', '(:': 'happy', '😀': 'happy', '😃': 'happy', '😄': 'happy',
            '😊': 'happy', '🙂': 'happy', '❤️': 'happy',
            ':(': 'sad', ':-(': 'sad', '😢': 'sad', '😭': 'sad', '😞': 'sad', '☹️': 'sad',
            '>:(': 'angry', '😠': 'angry', '😡': 'angry',
            ':o': 'surprise', ':-o': 'surprise', '😮': 'surprise', '😲': 'surprise', '😳': 'surprise',
            '🤢': 'disgust', '🤮': 'disgust',
            '😨': 'fear', '😱': 'fear'
        }

        # Normalize negations to forms without apostrophes (we'll normalize tokens too)
        self.negations = {"not","no","never","dont","doesnt","didnt","cant","wont","isnt","arent","couldnt","hardly","barely","rarely"}

        # modifiers
        self.intensifiers = {"very":1.5,"extremely":1.8,"so":1.4,"really":1.3,"super":1.5,"too":1.2,"incredibly":1.8,"quite":1.2,"totally":1.4}
        self.diminishers = {"slightly":0.7,"somewhat":0.8,"abit":0.8,"little":0.85,"kinda":0.85,"kindof":0.85}

        # opposite mapping for negation flip
        self.opposite = {'happy':'sad','sad':'happy','angry':'happy','fear':'happy','disgust':'happy','surprise':'neutral'}

        # simple phrase patterns (multi-word matches)
        self.phrases = {
            'happy': [re.compile(r'\bgood job\b', re.I), re.compile(r'\bwell done\b', re.I)],
            'surprise': [re.compile(r"\bwhat a surprise\b", re.I), re.compile(r"\bdidn't expect\b", re.I)]
        }

    def _tokenize(self, text: str) -> List[str]:
        # Normalize: lower-case and remove odd punctuation but keep emoticon symbols and ! ?.
        cleaned = re.sub(r"[^\w\s!?:;()<>:\-]+", " ", text.lower())
        tokens = re.findall(r"\w+|!+|\?+|[:;]-?[)(oOpP]|[()<>]", cleaned)
        return tokens

    def _score(self, text: str) -> Dict[str, float]:
        if not text:
            return {k:0.0 for k in self.lex.keys()}

        text_norm = text.lower()
        scores = {k:0.0 for k in self.lex.keys()}

        # Emoji/emoticon matches (use raw text, emojis are non-ascii)
        for emj, emo in self.emoji_map.items():
            if emj in text:
                scores[emo] += 1.0

        # Phrase matches (multiword)
        for emo, pats in self.phrases.items():
            for p in pats:
                if p.search(text):
                    scores[emo] += 1.2

        tokens = self._tokenize(text_norm)

        # Word-level scan with windowed negation/modifiers
        for i, tok in enumerate(tokens):
            if not tok or tok in {'!','!!','!!!','?','??'}:
                continue

            # basic normalization for token comparisons
            tok_norm = tok.replace("'", "").replace("’", "")

            # Base match weight
            base_emo = None
            for emo, words in self.lex.items():
                if tok_norm in words:
                    base_emo = emo
                    break
            if not base_emo:
                continue

            weight = 1.0
            window = tokens[max(0, i-3):i]
            # handle modifiers
            for w in window:
                wn = w.replace("'", "")
                if wn in self.intensifiers:
                    weight *= self.intensifiers[wn]
                if wn in self.diminishers:
                    weight *= self.diminishers[wn]

            # Negation flipping
            if any(w.replace("'", "") in self.negations for w in window):
                opp = self.opposite.get(base_emo, 'neutral')
                if opp != 'neutral':
                    scores[opp] += 0.8 * weight
                weight *= 0.1

            scores[base_emo] += weight

        # Only boost with punctuation if we already have evidence
        if sum(scores.values()) > 0:
            exclam = text.count('!')
            if exclam >= 1:
                strongest = max(scores.items(), key=lambda x:x[1])[0]
                scores[strongest] += min(0.5, 0.1 * exclam)

        return scores

    def _ensure_hf(self) -> bool:
        """Initialize Hugging Face pipeline if available. Returns True if ready."""
        if self._hf_pipe is not None:
            return True
        if pipeline is None:
            return False
        try:
            # Compact, accurate English emotion model
            model_name = "j-hartmann/emotion-english-distilroberta-base"
            self._hf_pipe = pipeline("text-classification", model=model_name, return_all_scores=True, top_k=None)
            return True
        except Exception:
            # If model fails to load (e.g., offline), skip HF
            self._hf_pipe = None
            return False

    def _analyze_transformer(self, text: str) -> Optional[Tuple[str, float]]:
        """Use transformer model to get emotion -> (label, confidence). Returns None on failure."""
        if not text or not self._ensure_hf():
            return None
        try:
            outputs = self._hf_pipe(text[:2048])  # limit extremely long inputs
            # outputs is a list with one item: list of dicts {label, score}
            scores = outputs[0]
            # Map model labels to our schema
            label_map = {
                'joy': 'happy',
                'anger': 'angry',
                'sadness': 'sad',
                'fear': 'fear',
                'disgust': 'disgust',
                'surprise': 'surprise',
                # Optional labels commonly present
                'love': 'happy',
                'neutral': 'neutral'
            }
            best_label = None
            best_score = -1.0
            for item in scores:
                raw = item.get('label', '').lower()
                score = float(item.get('score', 0.0))
                mapped = label_map.get(raw)
                if mapped is None:
                    continue
                if score > best_score:
                    best_score = score
                    best_label = mapped
            if best_label is None:
                return None
            return best_label, max(0.0, min(1.0, best_score))
        except Exception:
            return None

    def analyze_text(self, text: str) -> Tuple[str, float]:
        text = (text or '').strip()
        if not text:
            return 'neutral', 0.0

        # Try transformer first
        hf_result = self._analyze_transformer(text)
        if hf_result is not None:
            return hf_result

        # Fallback to lexicon-based scoring
        scores = self._score(text)
        total = sum(scores.values())
        if total <= 0.1:
            return 'neutral', 0.0

        probs = {k: v/total for k, v in scores.items()}
        ordered = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        best, conf = ordered[0]
        # simple margin logic from original (optional)
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = conf - second
        if conf < 0.22 or margin < 0.06:
            # basic valence fallback (as you used)
            val_pos = scores.get('happy', 0.0) + 0.2 * scores.get('surprise', 0.0)
            val_neg = scores.get('sad', 0.0) + scores.get('angry', 0.0) + 0.3 * scores.get('fear', 0.0) + 0.3 * scores.get('disgust', 0.0)
            if val_pos - val_neg > 0.2:
                return 'happy', max(0.22, conf)
            if val_neg - val_pos > 0.2:
                neg_best = max({k: scores[k] for k in ['sad', 'angry', 'fear', 'disgust']}.items(), key=lambda x: x[1])[0]
                return neg_best, max(0.22, conf)
            return 'neutral', conf

        return best, conf
