"""
Main Integration Script
Integrates all components for the hybrid visual parser
"""

import time
import cv2
import numpy as np
from typing import Dict, Optional
import json

from region_calibrator import RegionCalibrator
from card_template_matcher import CardTemplateMatcher
from number_ocr import NumberOCR, MultiAttemptOCR
from state_tracker import TableStateManager
from stack_delta_inference import BetTracker
from hand_state_machine import HandStateMachine, HandState


class PokerTableParser:
    """
    Main parser that integrates all components:
    - Region calibration
    - Card template matching
    - OCR for numbers
    - State tracking with confirmation
    - Stack delta inference
    - Hand state machine validation
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        # Initialize components
        self.region_calibrator = RegionCalibrator()
        self.card_matcher = CardTemplateMatcher()
        self.ocr = NumberOCR(tesseract_path)
        self.multi_ocr = MultiAttemptOCR(self.ocr, attempts=3)
        self.state_manager = TableStateManager()
        self.bet_tracker = BetTracker()
        self.state_machine = HandStateMachine()
        
        # Configuration
        self.read_interval = 0.5  # Read every 0.5 seconds
        self.running = False
        
        # Load templates and regions
        self.templates_loaded = self.card_matcher.load_templates()
        self.regions_loaded = len(self.region_calibrator.load_regions()) > 0
    
    def calibrate(self):
        """Run region calibration"""
        print("=== Starting Region Calibration ===")
        self.region_calibrator.calibrate_all_regions()
        self.regions_loaded = True
        print("=== Calibration Complete ===")
    
    def load_calibration(self) -> bool:
        """Load existing calibration"""
        self.regions_loaded = len(self.region_calibrator.load_regions()) > 0
        return self.regions_loaded
    
    def create_card_templates(self):
        """Create card templates from screenshots"""
        from card_template_matcher import CardTemplateCreator
        creator = CardTemplateCreator()
        creator.create_all_templates()
        self.templates_loaded = self.card_matcher.load_templates()
    
    def read_table_once(self) -> Dict:
        """
        Read the table once and return raw data
        Returns dictionary with all detected values
        """
        if not self.regions_loaded:
            print("Error: No regions calibrated. Run calibrate() first.")
            return {}
        
        data = {}
        
        # Capture hero cards region
        hero_region = self.region_calibrator.capture_region("hero_cards")
        if hero_region is not None:
            hero_cards = self.card_matcher.extract_cards_from_zone(hero_region, max_cards=2)
            data['hero_cards'] = [card[0] for card in hero_cards] if hero_cards else []
        
        # Capture board cards region
        board_region = self.region_calibrator.capture_region("board_cards")
        if board_region is not None:
            board_cards = self.card_matcher.extract_cards_from_zone(board_region, max_cards=5)
            data['board_cards'] = [card[0] for card in board_cards] if board_cards else []
        
        # Capture pot size region
        pot_region = self.region_calibrator.capture_region("pot_size")
        if pot_region is not None:
            pot_size = self.multi_ocr.extract_dollar_amount_confirmed(pot_region)
            data['pot_size'] = pot_size
        
        # Capture action/bet region
        action_region = self.region_calibrator.capture_region("action_bet")
        if action_region is not None:
            bet_size = self.multi_ocr.extract_dollar_amount_confirmed(action_region)
            action_text = self.ocr.extract_timer_text(action_region)
            data['current_bet'] = bet_size
            data['action_text'] = action_text
        
        # Capture player stack regions
        for i in range(9):
            stack_key = f"player_{i}_stack" if i > 0 else "hero_stack"
            stack_region = self.region_calibrator.capture_region(stack_key)
            if stack_region is not None:
                stack_size = self.multi_ocr.extract_dollar_amount_confirmed(stack_region)
                data[stack_key] = stack_size
            else:
                break  # No more players
        
        # Capture dealer button region
        dealer_region = self.region_calibrator.capture_region("dealer_button")
        if dealer_region is not None:
            # Simple detection: check if button is visible (bright spot)
            gray = cv2.cvtColor(dealer_region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(thresh) > 0:
                data['dealer_visible'] = True
            else:
                data['dealer_visible'] = False
        
        return data
    
    def process_reading(self, data: Dict, timestamp: float):
        """
        Process a single reading through the state management system
        """
        # Update pot size with confirmation
        if 'pot_size' in data and data['pot_size'] is not None:
            self.state_manager.update_pot_size(data['pot_size'], 'ocr')
            self.bet_tracker.update_pot(data['pot_size'], timestamp)
        
        # Update hero cards with confirmation
        if 'hero_cards' in data:
            self.state_manager.update_hero_cards(data['hero_cards'], 'template')
        
        # Update board cards with confirmation
        if 'board_cards' in data:
            self.state_manager.update_board_cards(data['board_cards'], 'template')
        
        # Update stack sizes with confirmation and delta inference
        for key in data:
            if 'stack' in key and data[key] is not None:
                # Extract player ID
                if key == 'hero_stack':
                    player_id = 0
                else:
                    player_id = int(key.split('_')[1])
                
                # Update state manager
                self.state_manager.update_stack_size(player_id, data[key], 'ocr')
                
                # Update bet tracker for delta inference
                self.bet_tracker.update_player_stack(player_id, data[key], timestamp)
        
        # Update action text
        if 'action_text' in data and data['action_text'] is not None:
            self.state_manager.update_action_text(data['action_text'], 'ocr')
        
        # Update hand state machine
        hero_cards = data.get('hero_cards', [])
        board_cards = data.get('board_cards', [])
        pot_size = data.get('pot_size', 0.0)
        
        self.state_machine.validate_and_update(hero_cards, board_cards, pot_size)
    
    def get_confirmed_state(self) -> Dict:
        """Get the current confirmed state"""
        state = self.state_manager.get_state_snapshot()
        
        # Add hand state machine info
        hand_info = self.state_machine.get_hand_info()
        state['hand_state'] = hand_info.state.value
        state['hand_state_enum'] = hand_info.state
        
        # Add inferred bets from delta
        state['inferred_bets'] = self.bet_tracker.get_all_current_bets()
        
        return state
    
    def run_once(self) -> Dict:
        """Run a single detection cycle"""
        timestamp = time.time()
        data = self.read_table_once()
        self.process_reading(data, timestamp)
        return self.get_confirmed_state()
    
    def run_continuous(self, duration: Optional[float] = None):
        """
        Run continuous detection
        Args:
            duration: Run for specified seconds, or None to run indefinitely
        """
        self.running = True
        start_time = time.time()
        
        print("=== Starting Continuous Detection ===")
        print(f"Reading interval: {self.read_interval}s")
        if duration:
            print(f"Duration: {duration}s")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Run detection cycle
                state = self.run_once()
                
                # Print state summary
                self._print_state_summary(state)
                
                # Wait for next interval
                time.sleep(self.read_interval)
        
        except KeyboardInterrupt:
            print("\n=== Stopped by user ===")
        
        self.running = False
        print("=== Detection Stopped ===")
    
    def stop(self):
        """Stop continuous detection"""
        self.running = False
    
    def _print_state_summary(self, state: Dict):
        """Print a summary of current state"""
        hand_state = state.get('hand_state', 'unknown')
        pot_size = state.get('pot_size', 'N/A')
        hero_cards = state.get('hero_cards', [])
        board_cards = state.get('board_cards', [])
        
        print(f"[{hand_state.upper()}] Pot: ${pot_size:.2f} | Hero: {hero_cards} | Board: {board_cards}")
    
    def export_state(self, filepath: str):
        """Export current state to JSON file"""
        state = self.get_confirmed_state()
        
        # Convert enum to string for JSON serialization
        if 'hand_state_enum' in state:
            del state['hand_state_enum']
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"State exported to {filepath}")
    
    def get_status(self) -> Dict:
        """Get system status"""
        return {
            'templates_loaded': self.templates_loaded,
            'regions_loaded': self.regions_loaded,
            'running': self.running,
            'card_templates': len(self.card_matcher.templates),
            'calibrated_regions': len(self.region_calibrator.regions),
        }


def main():
    """Main entry point"""
    import sys
    
    parser = PokerTableParser()
    
    # Check status
    status = parser.get_status()
    print("=== Poker Table Parser Status ===")
    print(f"Templates loaded: {status['templates_loaded']}")
    print(f"Regions calibrated: {status['regions_loaded']}")
    print()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "calibrate":
            parser.calibrate()
        
        elif command == "create_templates":
            parser.create_card_templates()
        
        elif command == "test":
            # Run single detection test
            if not status['regions_loaded']:
                print("No regions calibrated. Run 'python main.py calibrate' first.")
                return
            if not status['templates_loaded']:
                print("No card templates loaded. Run 'python main.py create_templates' first.")
                return
            
            state = parser.run_once()
            print("\n=== Detection Result ===")
            print(json.dumps(state, indent=2, default=str))
        
        elif command == "run":
            # Run continuous detection
            if not status['regions_loaded']:
                print("No regions calibrated. Run 'python main.py calibrate' first.")
                return
            if not status['templates_loaded']:
                print("No card templates loaded. Run 'python main.py create_templates' first.")
                return
            
            duration = float(sys.argv[2]) if len(sys.argv) > 2 else None
            parser.run_continuous(duration)
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: calibrate, create_templates, test, run [duration]")
    
    else:
        print("Usage:")
        print("  python main.py calibrate           - Calibrate screen regions")
        print("  python main.py create_templates    - Create card templates")
        print("  python main.py test                - Run single detection test")
        print("  python main.py run [duration]      - Run continuous detection")


if __name__ == "__main__":
    main()
