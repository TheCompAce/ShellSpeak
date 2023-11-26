# Save settings to a JSON file
import json
from modules.save_load import load_settings, save_settings

# Setup menu function
def setup_menu():
    # Define settings file path
    settings_filepath = 'settings.json'
    
    # Load existing settings
    settings = load_settings(settings_filepath)

    # Setup menu loop
    while True:
        print("\nSetup Menu:")
        print("1. System Prompt")
        print("2. User Prompt")
        print("3. Display Prompt")
        print(f"4. Use Indexing (Currently {'Enabled' if settings.get('use_indexing', False) else 'Disabled'})")
        print("5. Back to Main Menu")
        
        choice = input("Choose an option: ")
        
        if choice == '1':
            settings['command_prompt'] = input("Enter the System Prompt (or File Path): ")
        elif choice == '2':
            settings['user_command_prompt'] = input("Enter the User Prompt (or File Path): ")
        elif choice == '3':
            settings['display_prompt'] = input("Enter the Display Prompt (or File Path): ")        
        elif choice == '4':
            current_status = 'enabled' if settings.get('use_indexing', False) else 'disabled'
            check = input(f"Do you want to enable Index Building? It is currently {current_status}. (yes/no): ")
            settings['use_indexing'] = check.lower() in ["yes", "y"]
        elif choice == '5':
            print("Returning to Main Menu.")
            break
        else:
            print("Invalid option. Please try again.")
        
        # Save updated settings
        save_settings(settings, settings_filepath)