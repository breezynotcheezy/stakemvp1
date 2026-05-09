# Poker Table Visual Parser

Hybrid visual parser for reliable poker table detection using automatic card recognition, template matching, and OCR.

## Architecture

1. **Screen Region Calibration** - User selects fixed zones once
2. **Automatic Card Recognition** - AI detects cards using color detection and OCR (no templates required!)
3. **Optional Card Templates** - Manual templates for improved accuracy (optional)
4. **OCR for Numbers** - Pot, stack, and bet sizes only
5. **State Tracking** - Time-based confirmation (2-3 readings)
6. **Stack Delta Inference** - Calculate bets from stack changes
7. **Hand State Machine** - Validate against poker rules

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python calibrate_regions.py  # First: calibrate screen regions
python main.py run           # Run the parser (auto-recognition enabled by default)
```

**Optional:** Create manual card templates for improved accuracy
```bash
python create_card_templates.py  # Optional: create card templates from screenshots
```

## Automatic Card Recognition

The system now automatically recognizes cards without requiring manual templates:

- **Suit Detection**: Uses HSV color analysis to detect red (hearts/diamonds) vs black (clubs/spades)
- **Rank Detection**: Uses OCR to read the rank character (A, K, Q, J, T, 9, 8, etc.) from card corners
- **Shape Analysis**: Uses contour detection to distinguish between similar suit symbols
- **Confidence Scoring**: Provides confidence scores for each recognition

## Requirements

- Tesseract OCR must be installed and in PATH
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
