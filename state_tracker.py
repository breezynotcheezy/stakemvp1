"""
State Tracking Over Time
Tracks table state over time with confirmation to avoid OCR mistakes
"""

import time
from typing import Dict, Optional, List, Any
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StateReading:
    """A single reading of a state value"""
    value: Any
    timestamp: float
    source: str


class TimeBasedStateTracker:
    """
    Tracks state values over time and requires confirmation
    before accepting a value as valid
    """
    
    def __init__(self, confirmation_count: int = 3, confirmation_window: float = 2.0):
        """
        Args:
            confirmation_count: Number of identical readings required
            confirmation_window: Time window in seconds for confirmation
        """
        self.confirmation_count = confirmation_count
        self.confirmation_window = confirmation_window
        
        # Store readings for each state variable
        self.readings: Dict[str, deque] = {}
        
        # Current confirmed values
        self.confirmed_values: Dict[str, Any] = {}
        
        # Last update timestamp for each value
        self.last_update: Dict[str, float] = {}
    
    def add_reading(self, key: str, value: Any, source: str = "unknown") -> Optional[Any]:
        """
        Add a new reading for a state variable
        Returns the confirmed value if confirmation threshold reached
        """
        timestamp = time.time()
        reading = StateReading(value, timestamp, source)
        
        # Initialize deque for this key if needed
        if key not in self.readings:
            self.readings[key] = deque(maxlen=self.confirmation_count * 2)
        
        self.readings[key].append(reading)
        
        # Clean old readings outside confirmation window
        self._clean_old_readings(key, timestamp)
        
        # Check for confirmation
        return self._check_confirmation(key)
    
    def _clean_old_readings(self, key: str, current_time: float):
        """Remove readings outside the confirmation window"""
        if key not in self.readings:
            return
        
        while (self.readings[key] and 
               current_time - self.readings[key][0].timestamp > self.confirmation_window):
            self.readings[key].popleft()
    
    def _check_confirmation(self, key: str) -> Optional[Any]:
        """
        Check if we have enough identical readings to confirm a value
        Returns confirmed value if threshold reached
        """
        if key not in self.readings or len(self.readings[key]) < self.confirmation_count:
            return None
        
        # Count occurrences of each value
        value_counts: Dict[Any, int] = {}
        for reading in self.readings[key]:
            val = reading.value
            if val is not None:
                value_counts[val] = value_counts.get(val, 0) + 1
        
        # Find value with enough confirmations
        for value, count in value_counts.items():
            if count >= self.confirmation_count:
                # Only update if value changed
                if key not in self.confirmed_values or self.confirmed_values[key] != value:
                    self.confirmed_values[key] = value
                    self.last_update[key] = time.time()
                    print(f"[StateTracker] Confirmed {key} = {value}")
                return value
        
        return None
    
    def get_confirmed_value(self, key: str) -> Optional[Any]:
        """Get the current confirmed value for a key"""
        return self.confirmed_values.get(key)
    
    def get_all_confirmed_values(self) -> Dict[str, Any]:
        """Get all confirmed values"""
        return self.confirmed_values.copy()
    
    def reset_key(self, key: str):
        """Reset tracking for a specific key"""
        if key in self.readings:
            self.readings[key].clear()
        if key in self.confirmed_values:
            del self.confirmed_values[key]
        if key in self.last_update:
            del self.last_update[key]
    
    def reset_all(self):
        """Reset all tracking"""
        self.readings.clear()
        self.confirmed_values.clear()
        self.last_update.clear()
    
    def is_value_stale(self, key: str, stale_threshold: float = 5.0) -> bool:
        """
        Check if a confirmed value is stale (not updated recently)
        Args:
            stale_threshold: Time in seconds before value is considered stale
        """
        if key not in self.last_update:
            return True
        return time.time() - self.last_update[key] > stale_threshold


class TableStateManager:
    """
    Manages the entire poker table state with time-based confirmation
    """
    
    def __init__(self):
        self.tracker = TimeBasedStateTracker(confirmation_count=3, confirmation_window=1.5)
        
        # State categories
        self.state_categories = {
            'pot_size': 'numeric',
            'hero_stack': 'numeric',
            'player_0_stack': 'numeric',
            'player_1_stack': 'numeric',
            'player_2_stack': 'numeric',
            'player_3_stack': 'numeric',
            'player_4_stack': 'numeric',
            'player_5_stack': 'numeric',
            'player_6_stack': 'numeric',
            'player_7_stack': 'numeric',
            'player_8_stack': 'numeric',
            'current_bet': 'numeric',
            'hero_cards': 'list',
            'board_cards': 'list',
            'action_text': 'string',
            'timer': 'numeric',
            'dealer_position': 'numeric',
        }
    
    def update_pot_size(self, value: float, source: str = "ocr") -> Optional[float]:
        """Update pot size with confirmation"""
        return self.tracker.add_reading('pot_size', value, source)
    
    def update_stack_size(self, player_id: int, value: float, source: str = "ocr") -> Optional[float]:
        """Update player stack size with confirmation"""
        key = f'player_{player_id}_stack' if player_id > 0 else 'hero_stack'
        return self.tracker.add_reading(key, value, source)
    
    def update_current_bet(self, value: float, source: str = "ocr") -> Optional[float]:
        """Update current bet size with confirmation"""
        return self.tracker.add_reading('current_bet', value, source)
    
    def update_hero_cards(self, cards: List[str], source: str = "template") -> Optional[List[str]]:
        """Update hero cards with confirmation"""
        # For lists, we need to compare as tuples for hashing
        cards_tuple = tuple(sorted(cards))
        result = self.tracker.add_reading('hero_cards', cards_tuple, source)
        if result:
            return list(result)
        return None
    
    def update_board_cards(self, cards: List[str], source: str = "template") -> Optional[List[str]]:
        """Update board cards with confirmation"""
        cards_tuple = tuple(sorted(cards))
        result = self.tracker.add_reading('board_cards', cards_tuple, source)
        if result:
            return list(result)
        return None
    
    def update_action_text(self, text: str, source: str = "ocr") -> Optional[str]:
        """Update action text with confirmation"""
        return self.tracker.add_reading('action_text', text, source)
    
    def update_timer(self, value: float, source: str = "ocr") -> Optional[float]:
        """Update timer with confirmation"""
        return self.tracker.add_reading('timer', value, source)
    
    def update_dealer_position(self, position: int, source: str = "detection") -> Optional[int]:
        """Update dealer button position with confirmation"""
        return self.tracker.add_reading('dealer_position', position, source)
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get current confirmed state snapshot"""
        state = self.tracker.get_all_confirmed_values()
        
        # Convert tuples back to lists for card data
        if 'hero_cards' in state and isinstance(state['hero_cards'], tuple):
            state['hero_cards'] = list(state['hero_cards'])
        if 'board_cards' in state and isinstance(state['board_cards'], tuple):
            state['board_cards'] = list(state['board_cards'])
        
        return state
    
    def is_state_complete(self) -> bool:
        """Check if we have enough confirmed state to make decisions"""
        state = self.get_state_snapshot()
        
        # Minimum required state
        required_keys = ['pot_size', 'hero_stack', 'hero_cards']
        
        for key in required_keys:
            if key not in state:
                return False
        
        return True
    
    def get_stale_keys(self, threshold: float = 5.0) -> List[str]:
        """Get list of keys with stale values"""
        stale = []
        for key in self.state_categories.keys():
            if self.tracker.is_value_stale(key, threshold):
                stale.append(key)
        return stale
    
    def reset_hand_state(self):
        """Reset state for new hand"""
        # Keep stack sizes, reset cards and pot
        stack_keys = [k for k in self.state_categories.keys() if 'stack' in k]
        
        for key in list(self.tracker.confirmed_values.keys()):
            if 'stack' not in key:
                self.tracker.reset_key(key)


if __name__ == "__main__":
    # Test the state tracker
    tracker = TimeBasedStateTracker(confirmation_count=3, confirmation_window=1.5)
    
    print("Testing state tracker...")
    
    # Simulate readings
    for i in range(5):
        result = tracker.add_reading('pot_size', 100.0, 'ocr')
        print(f"Reading {i+1}: {result}")
        time.sleep(0.1)
    
    print(f"Final confirmed value: {tracker.get_confirmed_value('pot_size')}")
