import json
import os
import sys

from modules.menus.setup_menu import setup_menu
from modules.shellSpeak import ShellSpeak
from modules.utils import load_settings




def main():
    settings = load_settings('settings.json')

    # Check for command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '/start':
        shellSpeak = ShellSpeak(settings)
        shellSpeak.run()
        return
    
    # Display menu
    while True:
        print("\nMenu:")
        print("1. Setup")
        print("2. Run")
        print("3. Exit")
        choice = input("Choose an option: ")
        
        if choice == '1':
            setup_menu()
        elif choice == '2':
            shellSpeak = ShellSpeak(settings)
            shellSpeak.run()
            break
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
