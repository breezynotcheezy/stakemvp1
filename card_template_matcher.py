"""
Card Template Matching System
Uses image template matching instead of OCR for card detection
"""

import cv2
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class CardTemplateMatcher:
    """Matches cards using template matching instead of OCR"""
    
    # Card suits and ranks
    SUITS = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
    RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    
    def __init__(self, templates_dir: str = "card_templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        self.templates: Dict[str, np.ndarray] = {}
        self.match_threshold = 0.8
        
    def load_templates(self) -> bool:
        """Load all card templates from directory"""
        if not self.templates_dir.exists():
            print(f"Templates directory not found: {self.templates_dir}")
            return False
        
        loaded_count = 0
        for suit in self.SUITS:
            for rank in self.RANKS:
                card_name = f"{rank}{suit}"
                template_path = self.templates_dir / f"{card_name}.png"
                
                if template_path.exists():
                    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                    if template is not None:
                        self.templates[card_name] = template
                        loaded_count += 1
        
        print(f"Loaded {loaded_count} card templates")
        return loaded_count > 0
    
    def save_template(self, card_name: str, image: np.ndarray):
        """Save a card template image"""
        template_path = self.templates_dir / f"{card_name}.png"
        cv2.imwrite(str(template_path), image)
        print(f"Saved template: {card_name}")
    
    def match_card(self, card_image: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Match a card image against templates
        Returns (card_name, confidence) or None if no match
        """
        if not self.templates:
            if not self.load_templates():
                return None
        
        best_match = None
        best_confidence = 0.0
        
        for card_name, template in self.templates.items():
            # Resize template to match card image if needed
            if template.shape != card_image.shape:
                template = cv2.resize(template, (card_image.shape[1], card_image.shape[0]))
            
            # Template matching
            result = cv2.matchTemplate(card_image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_confidence:
                best_confidence = max_val
                best_match = card_name
        
        if best_confidence >= self.match_threshold:
            return best_match, best_confidence
        
        return None
    
    def extract_cards_from_zone(self, zone_image: np.ndarray, max_cards: int = 2) -> List[Tuple[str, float]]:
        """
        Extract and match multiple cards from a zone image
        Returns list of (card_name, confidence) tuples
        """
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(zone_image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to separate cards from background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio (cards are rectangular)
        card_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Skip small contours
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Cards typically have aspect ratio around 0.6-0.8
            if 0.5 < aspect_ratio < 1.0:
                card_contours.append((contour, (x, y, w, h)))
        
        # Sort by x position (left to right)
        card_contours.sort(key=lambda c: c[1][0])
        
        # Extract and match cards
        detected_cards = []
        for contour, (x, y, w, h) in card_contours[:max_cards]:
            card_image = zone_image[y:y+h, x:x+w]
            match = self.match_card(card_image)
            if match:
                detected_cards.append(match)
        
        return detected_cards
    
    def set_match_threshold(self, threshold: float):
        """Set the confidence threshold for matching"""
        self.match_threshold = max(0.0, min(1.0, threshold))
    
    def get_template_info(self) -> Dict:
        """Get information about loaded templates"""
        return {
            "total_templates": len(self.templates),
            "templates_dir": str(self.templates_dir),
            "match_threshold": self.match_threshold,
            "available_cards": list(self.templates.keys())
        }


class CardTemplateCreator:
    """Helper class to create card templates from screenshots"""
    
    def __init__(self, templates_dir: str = "card_templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        self.current_card = None
        self.drawing = False
        self.start_point = None
        self.image = None
        self.temp_image = None
    
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param):
        """Mouse callback for selecting card region"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_point:
                self.temp_image = self.image.copy()
                cv2.rectangle(self.temp_image, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("Card Selector", self.temp_image)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point:
                end_point = (x, y)
                x1, y1 = self.start_point
                x2, y2 = end_point
                
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                self.current_card = self.image[y1:y2, x1:x2]
                self.start_point = None
    
    def create_template_from_screen(self, card_name: str) -> bool:
        """Capture a card template from screen"""
        print(f"\nCreating template for card: {card_name}")
        print("Click and drag to select the card. Press ENTER to confirm, ESC to cancel.")
        
        # Capture screen
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            screenshot = sct.grab(monitor)
            self.image = np.array(screenshot)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR)
        
        self.temp_image = self.image.copy()
        self.current_card = None
        
        cv2.namedWindow("Card Selector")
        cv2.setMouseCallback("Card Selector", self.mouse_callback)
        
        while True:
            cv2.imshow("Card Selector", self.temp_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER
                if self.current_card is not None:
                    template_path = self.templates_dir / f"{card_name}.png"
                    cv2.imwrite(str(template_path), self.current_card)
                    print(f"Template saved: {card_name}")
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("No card selected!")
            
            elif key == 27:  # ESC
                print(f"Cancelled: {card_name}")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return False
    
    def create_all_templates(self):
        """Create templates for all 52 cards"""
        print("=== Card Template Creation ===")
        print("You will be asked to select each card from the screen.")
        print("Have a poker table visible with cards displayed.")
        
        created_count = 0
        for suit in CardTemplateMatcher.SUITS:
            for rank in CardTemplateMatcher.RANKS:
                card_name = f"{rank}{suit}"
                if self.create_template_from_screen(card_name):
                    created_count += 1
        
        print(f"\n=== Template Creation Complete ===")
        print(f"Created {created_count} card templates in {self.templates_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        # Create templates mode
        creator = CardTemplateCreator()
        creator.create_all_templates()
    else:
        # Test matching mode
        matcher = CardTemplateMatcher()
        if matcher.load_templates():
            print("Templates loaded successfully!")
            print(matcher.get_template_info())
        else:
            print("No templates found. Run with 'create' argument to create templates.")
