"""Test the new transformer-based text emotion analyzer"""

from text_emotion import TextEmotionAnalyzer

print("=" * 60)
print("Testing TextEmotionAnalyzer with Hugging Face Transformer")
print("=" * 60)

analyzer = TextEmotionAnalyzer()

# Test cases for all 7 emotions
test_cases = [
    ("I am so happy and excited about this amazing news!", "happy/joy"),
    ("This makes me really angry and frustrated!", "angry"),
    ("I'm feeling very sad and depressed today", "sad"),
    ("I'm scared and terrified about what might happen", "fear"),
    ("Wow, what an unexpected surprise!", "surprise"),
    ("I love you so much, you're wonderful!", "love/happy"),
    ("The weather is okay today.", "neutral"),
    ("Hello there", "neutral"),
    ("Just a normal day at work", "neutral"),
]

print("\nRunning tests...\n")
for text, expected in test_cases:
    emotion, confidence = analyzer.analyze_text(text)
    status = "✓" if expected.startswith(emotion) or emotion in expected else "?"
    print(f"{status} Text: '{text}'")
    print(f"  Expected: {expected} | Got: {emotion} (confidence: {confidence:.3f})\n")

print("=" * 60)
print("Test completed!")
print("=" * 60)
