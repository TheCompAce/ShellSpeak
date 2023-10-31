# Save settings to a JSON file
import json

# Load settings from a JSON file
def load_settings(filepath):
    try:
        with open(filepath, 'r') as f:
            settings = json.load(f)
        return settings
    except FileNotFoundError:
        return {}


def save_settings(settings, filepath):
    with open(filepath, 'w') as f:
        json.dump(settings, f, indent=4)

# Setup menu function
def setup_menu():
    # Define settings file path
    settings_filepath = 'settings.json'
    
    # Load existing settings
    settings = load_settings(settings_filepath)

    # Setup menu loop
    while True:
        print("\nSetup Menu:")
        print("1. Command Prompt")
        print("2. Display Prompt")
        print("3. Back to Main Menu")
        
        choice = input("Choose an option: ")
        
        if choice == '1':
            settings['command_prompt'] = input("Enter the Command Prompt (or File Path): ")
        elif choice == '2':
            settings['display_prompt'] = input("Enter the Display Prompt (or File Path): ")
        elif choice == '3':
            print("Returning to Main Menu.")
            break
        else:
            print("Invalid option. Please try again.")
        
        # Save updated settings
        save_settings(settings, settings_filepath)