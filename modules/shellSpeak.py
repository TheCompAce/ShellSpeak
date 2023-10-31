# Import necessary modules
import os
import subprocess
import logging

from modules.llm import LLM
from modules.utils import get_token_count, map_possible_commands, get_os_name, print_colored_text, capture_styled_input, trim_to_token_count, replace_placeholders

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ShellSpeak:
    def __init__(self, settings):
        self.llm = LLM(model_type=settings.get('model_type', 'OpenAI')) #Zephyr7bBeta
        self.settings = settings 

        self.llm_len = int(settings.get("llm_size", 948))

    def capture_input(self):
        # Get current working directory
        current_directory = os.getcwd()
        
        # Get environment (if available)
        environment = os.environ.get('VIRTUAL_ENV', None)
        if environment:
            environment = os.path.basename(environment)  # Extracting last part of the path as environment name
        
        # Formatted prompt
        prompt = f"/*c:green/*({environment})/*c:cyan/* {current_directory}/*c:white/*>" if environment else f"{current_directory}{self.settings['command_prompt']}"
        
        return capture_styled_input(prompt)

    def extract_shell_command(self, text):
        shell_section = text.split('```shell\n')[-1].split('```')[0].strip()
        return shell_section

    def execute_shell_section(self, shell_section):
        lines = shell_section.split('\n')
        if len(lines) == 1:
            # Single-line command, execute directly
            command = lines[0]
            return self.run_command(command)
        else:
            # Multi-line command, create a batch file
            batch_filename = 'temp.bat'
            if lines[0].startswith('#'):
                # Use commented out filename
                batch_filename = lines[0][1:].strip()
                lines = lines[1:]  # Remove the filename line
            with open(batch_filename, 'w') as batch_file:
                batch_file.write('\n'.join(lines))
            user_confirmation = input("Are you sure you want to run this batch file? (yes/no): ")
            if user_confirmation.lower() != 'yes':
                return "Canceled Command"
            output = self.run_command(batch_filename)
            if batch_filename == 'temp.bat':
                os.remove(batch_filename)  # Remove temporary batch file
            return output

    def run_command(self, command):
        result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout + result.stderr

    def translate_to_command(self, user_input):
        send_prompt = f'Use formating for reponse text [format]Utility function to print colored and styled text to the console. The text can contain special formatting tags like /*bold/*, and /*c:<color>/*. Example: "/*bold/*/*c:green/*Hello\n/*c:magenta/*/*nobold/*World!"[/format]'
        send_prompt = f"{send_prompt} {self.settings['display_prompt']}"        
        command_output = self.llm.ask(send_prompt, user_input)
        
        if '```shell' in command_output:
            shell_section = self.extract_shell_command(command_output)
            command_output = self.execute_shell_section(shell_section)
        return command_output

    def execute_command(self, command):
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
            if result.stderr:
                return False, result.stderr
            return True, result.stdout
        except Exception as e:
            return str(e)

    def translate_output(self, output):
        send_prompt = f'Use formating for reponse text [format]Utility function to print colored and styled text to the console. The text can contain special formatting tags like /*bold/*, and /*c:<color>/*. Example: "/*bold/*/*c:green/*Hello\n/*c:magenta/*/*nobold/*World!"[/format]'
        send_prompt = f"{send_prompt} {self.settings['display_prompt']}"
        display_output = self.llm.ask(send_prompt, output)
        return display_output

    def display_output(self, output):
        print_colored_text(output)

    def run(self):
        while True:
            user_input = self.capture_input()
            if user_input.lower() == 'exit':
                break
            logging.info(f"user_input = {user_input}")
            translated_command = self.translate_to_command(user_input)
            logging.info(f"translated_command = {translated_command}")
            sucess, raw_output = self.execute_command(translated_command)
            logging.info(f"raw_output = {raw_output}")
            translated_output = self.translate_output(raw_output)
            logging.info(f"translated_output = {translated_output}")
            self.display_output(translated_output)
