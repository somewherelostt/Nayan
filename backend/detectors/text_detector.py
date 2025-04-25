"""
Text detection module for Nayan
"""
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class TextDetector:
    def __init__(self):
        # OCR parameters
        self.tesseract_available = self._check_tesseract()
    
    def _check_tesseract(self):
        """Check if pytesseract is available"""
        try:
            import pytesseract
            return True
        except ImportError:
            print("Warning: pytesseract is not installed. OCR features will be disabled.")
            return False
    
    def perform_ocr(self, frame):
        """Extract text from the frame using Tesseract OCR"""
        if not self.tesseract_available:
            print("OCR requested but pytesseract is not available")
            return None
        
        try:
            import pytesseract
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhance image for better OCR results
            processed = self.enhance_image_for_ocr(gray)
            
            # Use Tesseract to extract text
            text = pytesseract.image_to_string(processed)
            
            # Clean and filter text
            if text.strip():
                filtered_text = ' '.join([line for line in text.split('\n') if len(line.strip()) > 3])
                if filtered_text:
                    return filtered_text
            return None
        except Exception as e:
            print(f"OCR error: {e}")
            return None
    
    def enhance_image_for_ocr(self, image):
        """Apply preprocessing to enhance image for better OCR results"""
        if len(image.shape) == 3:
            # Convert to grayscale if it's not already
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Apply dilation to make text more visible
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        return dilated 