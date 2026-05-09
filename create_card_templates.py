"""
Quick script to create card templates from screenshots
Run this after calibration to create templates for all 52 cards
"""

from card_template_matcher import CardTemplateCreator


def main():
    creator = CardTemplateCreator()
    creator.create_all_templates()


if __name__ == "__main__":
    main()
