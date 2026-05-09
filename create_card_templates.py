"""
Quick script to create card templates from screenshots
NOTE: Auto-recognition is now enabled by default - the AI figures out cards automatically
This script is OPTIONAL - only use if you want to improve accuracy with manual templates
"""

from card_template_matcher import CardTemplateCreator


def main():
    print("=== Card Template Creation ===")
    print("NOTE: Auto-recognition is enabled by default.")
    print("The AI will automatically recognize cards using color detection and OCR.")
    print("This script is OPTIONAL - use only if you want manual templates for improved accuracy.")
    print()
    
    response = input("Continue with manual template creation? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled. The system will use automatic card recognition.")
        return
    
    creator = CardTemplateCreator()
    creator.create_all_templates()


if __name__ == "__main__":
    main()
