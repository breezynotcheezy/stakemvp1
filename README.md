# Poker Table Visual Parser

Hybrid visual parser for reliable poker table detection using template matching and OCR.

## Architecture

1. **Screen Region Calibration** - User selects fixed zones once
2. **Card Template Matching** - Image templates for all 52 cards
3. **OCR for Numbers** - Pot, stack, and bet sizes only
4. **State Tracking** - Time-based confirmation (2-3 readings)
5. **Stack Delta Inference** - Calculate bets from stack changes
6. **Hand State Machine** - Validate against poker rules

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python calibrate_regions.py  # First: calibrate screen regions
python create_card_templates.py  # Second: create card templates from screenshots
python main.py  # Run the parser
```

## Requirements

- Tesseract OCR must be installed and in PATH
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
