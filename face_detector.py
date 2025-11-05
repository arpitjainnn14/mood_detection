import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # Load the pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect_faces(self, frame):
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            list: List of face locations (x, y, w, h)
            frame: Frame with face detection visualization
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangles around faces
        frame_with_faces = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return faces, frame_with_faces
    
    def extract_face(self, frame, face_location):
        """
        Extract face region from frame.
        
        Args:
            frame: Input frame
            face_location: Face location (x, y, w, h)
            
        Returns:
            numpy.ndarray: Extracted face image
        """
        x, y, w, h = face_location
        face_img = frame[y:y + h, x:x + w]
        return face_img
    
    def is_valid_face(self, face_img, min_size=30):
        """
        Check if the detected face is valid.
        
        Args:
            face_img: Face image
            min_size: Minimum face size
            
        Returns:
            bool: True if face is valid, False otherwise
        """
        if face_img is None or face_img.size == 0:
            return False
        
        height, width = face_img.shape[:2]
        if height < min_size or width < min_size:
            return False
        
        return True 