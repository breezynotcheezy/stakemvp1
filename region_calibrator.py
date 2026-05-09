"""
Screen Region Calibration System
Allows user to select fixed zones for poker table detection
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, Tuple, Optional
import pyautogui
import mss


class RegionCalibrator:
    """Calibrates screen regions for poker table detection"""
    
    def __init__(self, config_path: str = "regions.json"):
        self.config_path = config_path
        self.regions: Dict[str, Tuple[int, int, int, int]] = {}
        self.drawing = False
        self.current_region = None
        self.start_point = None
        self.image = None
        self.temp_image = None
        
    def load_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Load calibrated regions from config file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.regions = json.load(f)
            return self.regions
        return {}
    
    def save_regions(self):
        """Save calibrated regions to config file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.regions, f, indent=2)
        print(f"Regions saved to {self.config_path}")
    
    def capture_screen(self) -> np.ndarray:
        """Capture the entire screen"""
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # Primary monitor
            screenshot = sct.grab(monitor)
            image = np.array(screenshot)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return image
    
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param):
        """Mouse callback for drawing regions"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_point:
                self.temp_image = self.image.copy()
                cv2.rectangle(self.temp_image, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("Region Selector", self.temp_image)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point:
                end_point = (x, y)
                x1, y1 = self.start_point
                x2, y2 = end_point
                
                # Ensure coordinates are in correct order
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                self.current_region = (x1, y1, x2 - x1, y2 - y1)
                self.start_point = None
    
    def select_region(self, region_name: str, instruction: str) -> Optional[Tuple[int, int, int, int]]:
        """Interactive region selection"""
        print(f"\n{instruction}")
        print("Click and drag to select the region. Press ENTER to confirm, ESC to skip.")
        
        self.image = self.capture_screen()
        self.temp_image = self.image.copy()
        
        cv2.namedWindow("Region Selector")
        cv2.setMouseCallback("Region Selector", self.mouse_callback)
        
        while True:
            cv2.imshow("Region Selector", self.temp_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER
                if self.current_region:
                    self.regions[region_name] = self.current_region
                    print(f"Region '{region_name}' saved: {self.current_region}")
                    cv2.destroyAllWindows()
                    return self.current_region
                else:
                    print("No region selected!")
            
            elif key == 27:  # ESC
                print(f"Skipped region '{region_name}'")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        return None
    
    def calibrate_all_regions(self):
        """Calibrate all required poker table regions"""
        print("=== Poker Table Region Calibration ===")
        print("Make sure the poker table is visible on screen.")
        
        # Hero cards zone
        self.select_region(
            "hero_cards",
            "Select the HERO CARDS zone (your hole cards)"
        )
        
        # Board cards zone
        self.select_region(
            "board_cards",
            "Select the BOARD CARDS zone (community cards)"
        )
        
        # Pot size zone
        self.select_region(
            "pot_size",
            "Select the POT SIZE zone (total pot amount)"
        )
        
        # Player stack zones (up to 9 players)
        for i in range(9):
            region = self.select_region(
                f"player_{i}_stack",
                f"Select PLAYER {i+1} STACK zone (or press ESC to skip if not used)"
            )
            if region is None:
                break
        
        # Action/bet zone
        self.select_region(
            "action_bet",
            "Select the ACTION/BET zone (current bet/action text)"
        )
        
        # Dealer/button zone
        self.select_region(
            "dealer_button",
            "Select the DEALER BUTTON zone (dealer button position)"
        )
        
        self.save_regions()
        print("\n=== Calibration Complete ===")
        print(f"Calibrated regions: {list(self.regions.keys())}")
    
    def get_region(self, region_name: str) -> Optional[Tuple[int, int, int, int]]:
        """Get a specific region by name"""
        if not self.regions:
            self.load_regions()
        return self.regions.get(region_name)
    
    def capture_region(self, region_name: str) -> Optional[np.ndarray]:
        """Capture a specific region from screen"""
        region = self.get_region(region_name)
        if not region:
            return None
        
        x, y, width, height = region
        with mss.mss() as sct:
            monitor = {"top": y, "left": x, "width": width, "height": height}
            screenshot = sct.grab(monitor)
            image = np.array(screenshot)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return image


if __name__ == "__main__":
    calibrator = RegionCalibrator()
    calibrator.calibrate_all_regions()
