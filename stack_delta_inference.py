"""
Stack Delta Inference
Infers bets and actions from stack size changes over time
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Types of poker actions"""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"
    POST_BLIND = "post_blind"
    UNKNOWN = "unknown"


@dataclass
class StackChange:
    """Represents a change in stack size"""
    player_id: int
    previous_stack: float
    current_stack: float
    delta: float
    timestamp: float
    inferred_action: Optional[ActionType] = None
    inferred_amount: Optional[float] = None


class StackDeltaInference:
    """
    Infers player actions from stack size changes
    Stack deltas are often more reliable than reading bet labels directly
    """
    
    def __init__(self):
        # Track previous stack sizes
        self.previous_stacks: Dict[int, float] = {}
        
        # Track recent changes
        self.recent_changes: List[StackChange] = []
        
        # Track pot size changes to correlate with stack changes
        self.previous_pot: Optional[float] = None
        
        # Small blind and big blind amounts (can be updated dynamically)
        self.small_blind = 1.0
        self.big_blind = 2.0
    
    def update_stack(self, player_id: int, current_stack: float, timestamp: float) -> Optional[StackChange]:
        """
        Update stack size for a player and infer action if changed
        Returns StackChange if stack changed, None otherwise
        """
        if player_id not in self.previous_stacks:
            # First reading for this player
            self.previous_stacks[player_id] = current_stack
            return None
        
        previous_stack = self.previous_stacks[player_id]
        
        # Check if stack changed (with small tolerance for OCR noise)
        tolerance = 0.5  # $0.50 tolerance
        if abs(current_stack - previous_stack) < tolerance:
            return None
        
        # Calculate delta
        delta = current_stack - previous_stack
        
        # Update previous stack
        self.previous_stacks[player_id] = current_stack
        
        # Infer action from delta
        action, amount = self._infer_action_from_delta(delta, previous_stack, current_stack)
        
        # Create stack change record
        change = StackChange(
            player_id=player_id,
            previous_stack=previous_stack,
            current_stack=current_stack,
            delta=delta,
            timestamp=timestamp,
            inferred_action=action,
            inferred_amount=amount
        )
        
        self.recent_changes.append(change)
        
        # Keep only recent changes (last 20)
        if len(self.recent_changes) > 20:
            self.recent_changes.pop(0)
        
        return change
    
    def _infer_action_from_delta(self, delta: float, previous_stack: float, current_stack: float) -> Tuple[ActionType, Optional[float]]:
        """
        Infer action type and amount from stack delta
        Returns (action_type, amount)
        """
        if delta > 0:
            # Stack increased - won pot or received change
            return ActionType.UNKNOWN, None
        
        # Stack decreased - player put money in pot
        amount_lost = abs(delta)
        
        # Check for all-in
        if current_stack == 0:
            return ActionType.ALL_IN, amount_lost
        
        # Check for blind posting
        if abs(amount_lost - self.small_blind) < 0.5:
            return ActionType.POST_BLIND, amount_lost
        if abs(amount_lost - self.big_blind) < 0.5:
            return ActionType.POST_BLIND, amount_lost
        
        # Check for check (no change, but we already filtered that)
        if amount_lost < 0.1:
            return ActionType.CHECK, 0.0
        
        # Otherwise, it's a bet, call, or raise
        # We need more context to distinguish these
        return ActionType.BET, amount_lost
    
    def update_pot(self, current_pot: float, timestamp: float):
        """Update pot size for correlation with stack changes"""
        if self.previous_pot is not None:
            pot_delta = current_pot - self.previous_pot
            self._correlate_pot_with_stacks(pot_delta, timestamp)
        self.previous_pot = current_pot
    
    def _correlate_pot_with_stacks(self, pot_delta: float, timestamp: float):
        """
        Correlate pot changes with stack changes to refine action inference
        """
        if pot_delta <= 0:
            return
        
        # Find stack changes around the same time
        recent_changes = [c for c in self.recent_changes 
                         if abs(c.timestamp - timestamp) < 2.0]
        
        if not recent_changes:
            return
        
        # Calculate total stack loss
        total_stack_loss = sum(abs(c.delta) for c in recent_changes if c.delta < 0)
        
        # If pot delta matches stack losses, refine action types
        if abs(total_stack_loss - pot_delta) < 1.0:
            # Pot increase matches stack losses - these were bets/calls
            for change in recent_changes:
                if change.delta < 0:
                    if change.inferred_action == ActionType.BET:
                        # Could be call or raise depending on context
                        # For now, keep as bet
                        pass
    
    def set_blind_amounts(self, small_blind: float, big_blind: float):
        """Update blind amounts"""
        self.small_blind = small_blind
        self.big_blind = big_blind
    
    def get_recent_changes(self, player_id: Optional[int] = None) -> List[StackChange]:
        """
        Get recent stack changes
        If player_id is specified, only return changes for that player
        """
        if player_id is not None:
            return [c for c in self.recent_changes if c.player_id == player_id]
        return self.recent_changes.copy()
    
    def get_current_stacks(self) -> Dict[int, float]:
        """Get current stack sizes for all players"""
        return self.previous_stacks.copy()
    
    def reset_for_new_hand(self):
        """Reset tracking for new hand"""
        self.previous_stacks.clear()
        self.recent_changes.clear()
        self.previous_pot = None
    
    def infer_bet_from_stack_delta(self, player_id: int) -> Optional[float]:
        """
        Infer the bet amount for a player based on their most recent stack change
        Returns bet amount or None if no recent change
        """
        player_changes = self.get_recent_changes(player_id)
        if not player_changes:
            return None
        
        most_recent = player_changes[-1]
        if most_recent.delta < 0:
            return abs(most_recent.delta)
        return None


class BetTracker:
    """
    Tracks bets and infers bet amounts from stack changes
    """
    
    def __init__(self):
        self.delta_inference = StackDeltaInference()
        self.current_bets: Dict[int, float] = {}
        self.round_bets: Dict[int, float] = {}  # Bets in current betting round
    
    def update_player_stack(self, player_id: int, stack: float, timestamp: float) -> Optional[float]:
        """
        Update player stack and return inferred bet amount
        """
        change = self.delta_inference.update_stack(player_id, stack, timestamp)
        
        if change and change.delta < 0:
            bet_amount = abs(change.delta)
            self.current_bets[player_id] = bet_amount
            return bet_amount
        
        return None
    
    def update_pot(self, pot: float, timestamp: float):
        """Update pot size for correlation"""
        self.delta_inference.update_pot(pot, timestamp)
    
    def get_current_bet(self, player_id: int) -> Optional[float]:
        """Get current bet for a player"""
        return self.current_bets.get(player_id)
    
    def get_all_current_bets(self) -> Dict[int, float]:
        """Get all current bets"""
        return self.current_bets.copy()
    
    def reset_round(self):
        """Reset bets for new betting round"""
        self.round_bets = self.current_bets.copy()
        self.current_bets.clear()
    
    def reset_hand(self):
        """Reset for new hand"""
        self.delta_inference.reset_for_new_hand()
        self.current_bets.clear()
        self.round_bets.clear()


if __name__ == "__main__":
    # Test stack delta inference
    inference = StackDeltaInference()
    
    print("Testing stack delta inference...")
    
    # Simulate a hand
    timestamp = 0.0
    
    # Player posts small blind
    change = inference.update_stack(0, 99.0, timestamp)
    print(f"Player 0 posts SB: {change}")
    timestamp += 0.5
    
    # Player posts big blind
    change = inference.update_stack(1, 98.0, timestamp)
    print(f"Player 1 posts BB: {change}")
    timestamp += 0.5
    
    # Player 0 calls
    change = inference.update_stack(0, 98.0, timestamp)
    print(f"Player 0 calls: {change}")
    timestamp += 0.5
    
    # Player 1 checks
    change = inference.update_stack(1, 98.0, timestamp)
    print(f"Player 1 checks: {change}")
    
    print(f"\nCurrent stacks: {inference.get_current_stacks()}")
