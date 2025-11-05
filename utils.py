import os
import cv2
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['logs', 'screenshots']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def save_screenshot(frame, emotion=None):
    """Save a screenshot with timestamp and emotion label."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshots/screenshot_{timestamp}"
    if emotion:
        filename += f"_{emotion}"
    filename += ".jpg"
    cv2.imwrite(filename, frame)
    return filename

def log_emotion(emotion, confidence):
    """Log emotion data to CSV file."""
    timestamp = datetime.now()
    log_file = 'logs/emotion_log.csv'
    
    data = {
        'timestamp': [timestamp],
        'emotion': [emotion],
        'confidence': [confidence]
    }
    
    df = pd.DataFrame(data)
    
    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)

def generate_emotion_report():
    """Generate a report of emotion statistics."""
    log_file = 'logs/emotion_log.csv'
    if not os.path.exists(log_file):
        return None
    
    df = pd.read_csv(log_file)
    
    # Calculate emotion statistics
    emotion_counts = df['emotion'].value_counts()
    avg_confidence = df.groupby('emotion')['confidence'].mean()
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Emotion distribution pie chart
    plt.subplot(1, 2, 1)
    plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%')
    plt.title('Emotion Distribution')
    
    # Average confidence bar chart
    plt.subplot(1, 2, 2)
    avg_confidence.plot(kind='bar')
    plt.title('Average Confidence by Emotion')
    plt.ylabel('Confidence')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the report
    report_file = f'logs/emotion_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(report_file)
    plt.close()
    
    return report_file

def draw_emotion_box(frame, face_location, emotion, confidence):
    """Draw bounding box and emotion label on frame."""
    x, y, w, h = face_location
    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Prepare emotion text
    text = f"{emotion}: {confidence:.2f}"
    
    # Draw background rectangle for text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, (x, y - 30), (x + text_size[0], y), (0, 255, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame

def preprocess_face(face_img, target_size=(48, 48)):
    """Preprocess face image for emotion detection."""
    # Convert to grayscale
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize
    face_img = cv2.resize(face_img, target_size)
    
    # Normalize
    face_img = face_img.astype('float32') / 255.0
    
    # Reshape for model input
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    
    return face_img 