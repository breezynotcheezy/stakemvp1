"""
Hand State Machine
Tracks poker hand progression and validates against poker rules
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


class HandState(Enum):
    """States of a poker hand"""
    WAITING_FOR_HAND = "waiting_for_hand"
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    HAND_OVER = "hand_over"


class ValidationError(Exception):
    """Raised when state validation fails"""
    pass


@dataclass
class HandInfo:
    """Information about the current hand"""
    state: HandState
    hero_cards: List[str]
    board_cards: List[str]
    pot_size: float
    timestamp: float


class HandStateMachine:
    """
    State machine for tracking poker hand progression
    Validates state transitions against poker rules
    """
    
    def __init__(self):
        self.current_state = HandState.WAITING_FOR_HAND
        self.previous_state = None
        
        # Track hand data
        self.hero_cards: List[str] = []
        self.board_cards: List[str] = []
        self.pot_size: float = 0.0
        
        # Track state transitions for debugging
        self.state_history: List[Tuple[HandState, float]] = []
        
        # Validation rules
        self.valid_transitions = {
            HandState.WAITING_FOR_HAND: [HandState.PREFLOP],
            HandState.PREFLOP: [HandState.FLOP, HandState.HAND_OVER],
            HandState.FLOP: [HandState.TURN, HandState.HAND_OVER],
            HandState.TURN: [HandState.RIVER, HandState.HAND_OVER],
            HandState.RIVER: [HandState.SHOWDOWN, HandState.HAND_OVER],
            HandState.SHOWDOWN: [HandState.HAND_OVER],
            HandState.HAND_OVER: [HandState.WAITING_FOR_HAND, HandState.PREFLOP],
        }
    
    def transition_to(self, new_state: HandState, force: bool = False) -> bool:
        """
        Transition to a new state with validation
        Returns True if transition successful, False otherwise
        """
        # Validate transition
        if not force:
            if new_state not in self.valid_transitions.get(self.current_state, []):
                print(f"[State Machine] Invalid transition: {self.current_state} -> {new_state}")
                return False
        
        # Perform transition
        self.previous_state = self.current_state
        self.current_state = new_state
        
        # Record history
        self.state_history.append((new_state, datetime.now().timestamp()))
        
        print(f"[State Machine] Transition: {self.previous_state} -> {self.current_state}")
        
        # Reset data when entering waiting state
        if new_state == HandState.WAITING_FOR_HAND:
            self._reset_hand_data()
        
        return True
    
    def update_hero_cards(self, cards: List[str]) -> bool:
        """
        Update hero cards and validate state
        Returns True if update valid
        """
        if not cards:
            return False
        
        # Validate card format
        for card in cards:
            if not self._is_valid_card(card):
                print(f"[State Machine] Invalid card format: {card}")
                return False
        
        # Should have 2 hero cards in active hand states
        if self.current_state in [HandState.PREFLOP, HandState.FLOP, HandState.TURN, HandState.RIVER]:
            if len(cards) != 2:
                print(f"[State Machine] Expected 2 hero cards, got {len(cards)}")
                return False
        
        self.hero_cards = cards
        return True
    
    def update_board_cards(self, cards: List[str]) -> bool:
        """
        Update board cards and validate state
        Returns True if update valid, may trigger state transition
        """
        # Validate card format
        for card in cards:
            if not self._is_valid_card(card):
                print(f"[State Machine] Invalid card format: {card}")
                return False
        
        # Validate board card count based on state
        expected_counts = {
            HandState.WAITING_FOR_HAND: 0,
            HandState.PREFLOP: 0,
            HandState.FLOP: 3,
            HandState.TURN: 4,
            HandState.RIVER: 5,
            HandState.SHOWDOWN: 5,
            HandState.HAND_OVER: 0,
        }
        
        expected = expected_counts.get(self.current_state, 0)
        
        # Check if board card count change indicates state transition
        if len(cards) == 3 and self.current_state == HandState.PREFLOP:
            return self.transition_to(HandState.FLOP)
        elif len(cards) == 4 and self.current_state == HandState.FLOP:
            return self.transition_to(HandState.TURN)
        elif len(cards) == 5 and self.current_state == HandState.TURN:
            return self.transition_to(HandState.RIVER)
        
        # Validate current state
        if expected > 0 and len(cards) != expected:
            print(f"[State Machine] Board card count mismatch: expected {expected}, got {len(cards)}")
            # Don't return False - this might be a detection error, not invalid state
            # Just log the warning
        
        self.board_cards = cards
        return True
    
    def update_pot_size(self, pot: float) -> bool:
        """
        Update pot size
        Returns True if update valid
        """
        if pot is None:
            # Pot not detected yet, don't update
            return True
        
        if pot < 0:
            print(f"[State Machine] Invalid pot size: {pot}")
            return False
        
        self.pot_size = pot
        return True
    
    def detect_state_from_cards(self, hero_cards: List[str], board_cards: List[str]) -> Optional[HandState]:
        """
        Infer state from card data
        Returns inferred state or None if can't determine
        """
        if not hero_cards and not board_cards:
            return HandState.WAITING_FOR_HAND
        
        if hero_cards and not board_cards:
            return HandState.PREFLOP
        
        if len(board_cards) == 3:
            return HandState.FLOP
        elif len(board_cards) == 4:
            return HandState.TURN
        elif len(board_cards) == 5:
            return HandState.RIVER
        
        return None
    
    def validate_and_update(self, hero_cards: List[str], board_cards: List[str], 
                           pot_size: float) -> bool:
        """
        Validate all data and update state machine
        Returns True if all data valid
        """
        # Detect potential state from cards
        inferred_state = self.detect_state_from_cards(hero_cards, board_cards)
        
        if inferred_state and inferred_state != self.current_state:
            # Check if transition is valid
            if inferred_state in self.valid_transitions.get(self.current_state, []):
                self.transition_to(inferred_state, force=False)
            elif inferred_state == HandState.PREFLOP and self.current_state == HandState.HAND_OVER:
                # New hand starting
                self.transition_to(inferred_state, force=True)
        
        # Update data
        valid = True
        valid &= self.update_hero_cards(hero_cards)
        valid &= self.update_board_cards(board_cards)
        valid &= self.update_pot_size(pot_size)
        
        return valid
    
    def is_detection_valid(self, hero_cards: List[str], board_cards: List[str]) -> bool:
        """
        Validate that detected cards make sense for current state
        Returns True if detection appears valid
        """
        # Check for impossible transitions
        if len(self.board_cards) == 5 and len(board_cards) == 3:
            print("[State Machine] Invalid: board went from 5 to 3 cards")
            return False
        
        if len(self.board_cards) == 4 and len(board_cards) == 2:
            print("[State Machine] Invalid: board went from 4 to 2 cards")
            return False
        
        # Check for card duplication
        if len(board_cards) != len(set(board_cards)):
            print("[State Machine] Invalid: duplicate board cards")
            return False
        
        return True
    
    def get_hand_info(self) -> HandInfo:
        """Get current hand information"""
        return HandInfo(
            state=self.current_state,
            hero_cards=self.hero_cards.copy(),
            board_cards=self.board_cards.copy(),
            pot_size=self.pot_size,
            timestamp=datetime.now().timestamp()
        )
    
    def _reset_hand_data(self):
        """Reset hand data for new hand"""
        self.hero_cards = []
        self.board_cards = []
        self.pot_size = 0.0
    
    def _is_valid_card(self, card: str) -> bool:
        """Validate card format (e.g., 'Ah', 'Kd', '7s')"""
        if len(card) != 2:
            return False
        
        rank, suit = card[0], card[1]
        
        valid_ranks = {'A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'}
        valid_suits = {'h', 'd', 'c', 's'}
        
        return rank in valid_ranks and suit in valid_suits
    
    def force_state(self, state: HandState):
        """Force transition to a state (for testing or recovery)"""
        self.transition_to(state, force=True)
    
    def get_state_summary(self) -> Dict:
        """Get summary of current state"""
        return {
            "current_state": self.current_state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "hero_cards": self.hero_cards,
            "board_cards": self.board_cards,
            "pot_size": self.pot_size,
            "state_transitions": len(self.state_history)
        }


if __name__ == "__main__":
    # Test hand state machine
    machine = HandStateMachine()
    
    print("Testing hand state machine...")
    
    # Simulate a hand
    print(f"\nInitial state: {machine.current_state}")
    
    # Deal cards
    machine.update_hero_cards(['Ah', 'Ks'])
    machine.transition_to(HandState.PREFLOP)
    print(f"After deal: {machine.current_state}")
    
    # Flop
    machine.update_board_cards(['2h', '5d', '9c'])
    print(f"After flop: {machine.current_state}")
    
    # Turn
    machine.update_board_cards(['2h', '5d', '9c', 'Qh'])
    print(f"After turn: {machine.current_state}")
    
    # River
    machine.update_board_cards(['2h', '5d', '9c', 'Qh', '7s'])
    print(f"After river: {machine.current_state}")
    
    # Showdown
    machine.transition_to(HandState.SHOWDOWN)
    print(f"After showdown: {machine.current_state}")
    
    # Hand over
    machine.transition_to(HandState.HAND_OVER)
    print(f"After hand over: {machine.current_state}")
    
    print(f"\nState summary: {machine.get_state_summary()}")
