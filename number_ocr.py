"""
OCR for Numbers Only
Uses OCR to read pot size, stack sizes, bet sizes, and timer/action text
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import Optional, Tuple, List
from pathlib import Path


class NumberOCR:
    """OCR specifically for reading numbers from poker table"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize OCR
        tesseract_path: Optional path to tesseract executable
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Configure tesseract for number recognition
        self.number_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.$,'
        self.text_config = r'--oem 3 --psm 7'
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        
        # Resize for better OCR (tesseract works better with larger images)
        scale = 2
        resized = cv2.resize(denoised, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    def extract_number(self, image: np.ndarray) -> Optional[float]:
        """
        Extract a number from an image region
        Returns float value or None if failed
        """
        try:
            preprocessed = self.preprocess_image(image)
            
            # OCR with number-only whitelist
            text = pytesseract.image_to_string(preprocessed, config=self.number_config)
            
            # Clean text and extract number
            text = text.strip().replace(',', '').replace('$', '')
            
            # Find first number in text
            match = re.search(r'\d+\.?\d*', text)
            if match:
                return float(match.group())
            
            return None
        except Exception as e:
            print(f"OCR error: {e}")
            return None
    
    def extract_dollar_amount(self, image: np.ndarray) -> Optional[float]:
        """
        Extract a dollar amount from an image region
        Returns float value or None if failed
        """
        try:
            preprocessed = self.preprocess_image(image)
            
            # OCR with dollar sign allowed
            text = pytesseract.image_to_string(preprocessed, config=self.text_config)
            
            # Clean text
            text = text.strip().replace(',', '')
            
            # Find dollar amount pattern
            match = re.search(r'\$?(\d+\.?\d*)', text)
            if match:
                return float(match.group(1))
            
            return None
        except Exception as e:
            print(f"OCR error: {e}")
            return None
    
    def extract_text(self, image: np.ndarray) -> Optional[str]:
        """
        Extract text from an image region (for action/timer text)
        Returns string or None if failed
        """
        try:
            preprocessed = self.preprocess_image(image)
            
            # OCR for general text
            text = pytesseract.image_to_string(preprocessed, config=self.text_config)
            
            # Clean and return
            text = text.strip()
            if text:
                return text
            
            return None
        except Exception as e:
            print(f"OCR error: {e}")
            return None
    
    def extract_stack_size(self, image: np.ndarray) -> Optional[float]:
        """
        Extract stack size from player stack region
        Returns float value or None if failed
        """
        return self.extract_dollar_amount(image)
    
    def extract_pot_size(self, image: np.ndarray) -> Optional[float]:
        """
        Extract pot size from pot region
        Returns float value or None if failed
        """
        return self.extract_dollar_amount(image)
    
    def extract_bet_size(self, image: np.ndarray) -> Optional[float]:
        """
        Extract bet size from action/bet region
        Returns float value or None if failed
        """
        return self.extract_dollar_amount(image)
    
    def extract_timer_text(self, image: np.ndarray) -> Optional[str]:
        """
        Extract timer/action text (e.g., "Fold", "Call", "15s")
        Returns string or None if failed
        """
        return self.extract_text(image)


class MultiAttemptOCR:
    """Performs multiple OCR attempts and returns most common result"""
    
    def __init__(self, ocr: NumberOCR, attempts: int = 3):
        self.ocr = ocr
        self.attempts = attempts
    
    def extract_number_confirmed(self, image: np.ndarray) -> Optional[float]:
        """
        Extract number with confirmation across multiple attempts
        Returns value only if same result appears multiple times
        """
        results = []
        
        for _ in range(self.attempts):
            value = self.ocr.extract_number(image)
            if value is not None:
                results.append(value)
        
        if not results:
            return None
        
        # Find most common value (with tolerance for small differences)
        results.sort()
        
        # Group similar values (within 1% tolerance)
        groups = []
        for val in results:
            added = False
            for group in groups:
                if abs(val - group[0]) / max(group[0], 1) < 0.01:
                    group.append(val)
                    added = True
                    break
            if not added:
                groups.append([val])
        
        # Return value from largest group
        largest_group = max(groups, key=len)
        
        # Require at least 2/3 of attempts to agree
        if len(largest_group) >= self.attempts * 2 / 3:
            return sum(largest_group) / len(largest_group)
        
        return None
    
    def extract_dollar_amount_confirmed(self, image: np.ndarray) -> Optional[float]:
        """
        Extract dollar amount with confirmation across multiple attempts
        """
        results = []
        
        for _ in range(self.attempts):
            value = self.ocr.extract_dollar_amount(image)
            if value is not None:
                results.append(value)
        
        if not results:
            return None
        
        results.sort()
        
        # Group similar values
        groups = []
        for val in results:
            added = False
            for group in groups:
                if abs(val - group[0]) / max(group[0], 1) < 0.01:
                    group.append(val)
                    added = True
                    break
            if not added:
                groups.append([val])
        
        largest_group = max(groups, key=len)
        
        if len(largest_group) >= self.attempts * 2 / 3:
            return sum(largest_group) / len(largest_group)
        
        return None


def test_ocr():
    """Test OCR functionality"""
    ocr = NumberOCR()
    
    # Create a test image with text
    test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "$150.50", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    result = ocr.extract_dollar_amount(test_image)
    print(f"Test result: {result}")


if __name__ == "__main__":
    test_ocr()
