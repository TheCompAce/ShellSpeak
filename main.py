import json
import os
import sys
import asyncio
from modules.vectorDatabase import VectorDatabase
# from modules.vectors import load_faiss_index, build_and_save_faiss_index, load_index_data, needs_index_update
import json
from datetime import datetime

from modules.menus.setup_menu import save_settings, setup_menu
from modules.shellSpeak import ShellSpeak
from modules.utils import load_settings

def run_async_function(func, *args):
    asyncio.run(func(*args))

async def start_shell_speak(settings, base_path, vector_db):    
    await main_start(settings, base_path, vector_db)

async def main_start(settings, base_path, vector_db):
    # Initialize VectorDatabase here if needed globally
    
    
    shellSpeak = ShellSpeak(settings, base_path, vector_db)

    await shellSpeak.run()



def main():
    base_path = os.path.abspath(".")
    settings = load_settings(base_path)

    # FAISS Index check and build prompt
    if settings.get('use_indexing', False):
        system_folder_path = os.path.join(base_path, 'system')
        # history_json_path = os.path.join(system_folder_path, 'history.json')
        vector_db_path = os.path.join(system_folder_path, 'vector')

        vector_db = VectorDatabase(path=settings.get('vector_db_path', system_folder_path), 
                               name=settings.get('vector_db_name', vector_db_path))

        if not os.path.exists(system_folder_path):
            os.makedirs(system_folder_path)

        # Check if 'system' folder and 'history.json' exist
        # if not os.path.exists(system_folder_path) or not os.path.exists(history_json_path):
        #    settings['use_indexing'] = False
            
        if vector_db.needs_index_update():
            user_decision = input("A new index needs to be built. Do you want to build it now? (yes/no): ")
            if user_decision.lower() in ['yes', 'y']:
                print("Building index... (Grab a Coffee.)")
                vector_db.train_untrained_responses()
                print("Index built and saved successfully.")
                settings['last_build_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                save_settings(settings, os.path.join(base_path, 'settings.json'))
            else:
                print("Skipping index building.")

    # Check for command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '/start':
        run_async_function(start_shell_speak, settings, base_path, vector_db)
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
            run_async_function(start_shell_speak, settings, base_path, vector_db)
            break
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
