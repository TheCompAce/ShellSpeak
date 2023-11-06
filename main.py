import json
import os
import sys
import asyncio

from modules.menus.setup_menu import setup_menu
from modules.shellSpeak import ShellSpeak
from modules.utils import load_settings

def run_async_function(func, *args):
    asyncio.run(func(*args))

async def start_shell_speak(settings, base_path):    
    await main_start(settings, base_path)

async def main_start(settings, base_path):
    shellSpeak = ShellSpeak(settings, base_path)
    await shellSpeak.run()

def main():
    base_path = os.path.abspath(".")
    settings = load_settings(base_path)

    # Check for command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '/start':
        run_async_function(start_shell_speak, settings, base_path)
        return
    
    # Display menu
    while True:
        print("\nMenu:")
        print("1. Setup")
        print("2. Run")
        print("3. Exit")
        print("-----------------------------------------------------------------")
        print("(You can also start the script with /start to Start automaticly.)")
        print("-----------------------------------------------------------------")
        choice = input("Choose an option: ")
        
        if choice == '1':
            setup_menu()
        elif choice == '2':
            run_async_function(start_shell_speak, settings, base_path)
            break
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
