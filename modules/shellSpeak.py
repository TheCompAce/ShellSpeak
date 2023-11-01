# Import necessary modules
import os
import platform
import re
import subprocess
import logging
import signal
import threading
import ctypes

from modules.llm import LLM, ModelTypes
from modules.utils import get_token_count, load_settings, map_possible_commands, get_os_name, print_colored_text, capture_styled_input, trim_to_token_count, replace_placeholders

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CommandResult:
    def __init__(self, stdout, stderr):
        self.out = stdout
        self.err = stderr

class ShellSpeak:
    def __init__(self, settings):
        self.llm_len = int(settings.get("llm_size", 500))
        self.llm_output_size = int(settings.get("llm_output_size", 700))
        self.use_cache = settings.get("use_cache", True)
        self.cache_file = settings.get("cache_file", None)
        self.settings = settings
        self.settingsRoot = os.path.abspath("settings.json")

        self.llm = LLM(model_type=self.settings.get('model_type', ModelTypes('OpenAI')), use_cache=self.use_cache, cache_file=self.cache_file) #Zephyr7bBeta

        logging.info(f"Shell Speak Loaded")

    def capture_input(self):
        # Get current working directory
        current_directory = os.getcwd()
        
        # Get environment (if available)
        environment = os.environ.get('VIRTUAL_ENV', None)
        if environment:
            environment = os.path.basename(environment)  # Extracting last part of the path as environment name
        
        # Formatted prompt
        prompt = f"[green]({environment})[cyan] {current_directory}[white]>" if environment else f"{current_directory}{self.settings['command_prompt']}"
        
        set_input = capture_styled_input(prompt)
        logging.info(f"Using input : {set_input}")
        return set_input
    
    def extract_python_command(self, text):
        match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if match:
            shell_section = match.group(1).strip()
        else:
            logging.error("No shell section found")
            shell_section = None
        return shell_section

    def execute_python_script(self, python_section):
        lines = python_section.split('\n')
        if len(lines) == 1:
            # Single-line script, execute directly
            script = lines[0]
            return self.run_python_script(script)
        else:
            # Multi-line script, create a python file
            python_filename = 'temp.py'
            if lines[0].startswith('#'):
                # Use commented out filename
                python_filename = lines[0][1:].strip()
                lines = lines[1:]  # Remove the filename line

            with open(python_filename, 'w') as python_file:
                python_file.write('\n'.join(lines))
            print("==== Python File ====")
            print('\n'.join(lines))
            print("====================")
            user_confirmation = input("Are you sure you want to run this Python script? (yes/no): ")
            if user_confirmation.lower() != 'yes':
                return "Canceled Command"
            output = self.run_python_script(python_filename)
            if python_filename == 'temp.py':
                os.remove(python_filename)  # Remove temporary python file
            return output
        
    def run_python_script(self, script):
        # If the script is a file, use 'python filename.py' to execute
        if script.endswith('.py'):
            command = f'python {script}'
        else:
            command = f'python -c "{script}"'
        result = self.run_command(command)
        return result.out + result.err

    def extract_shell_command(self, text):
        match = re.search(r'```shell(.*?)```', text, re.DOTALL)
        if match:
            shell_section = match.group(1).strip()
        else:
            logging.error("No shell section found")
            shell_section = None
        return shell_section
    
    def extract_batch_command(self, text):
        match = re.search(r'```batch(.*?)```', text, re.DOTALL)
        if match:
            shell_section = match.group(1).strip()
        else:
            logging.error("No shell section found")
            shell_section = None
        return shell_section

    def extract_bash_command(self, text):
        match = re.search(r'```bash(.*?)```', text, re.DOTALL)
        if match:
            shell_section = match.group(1).strip()
        else:
            logging.error("No shell section found")
            shell_section = None
        return shell_section

    def execute_shell_section(self, shell_section):

        logging.info(f"Executing Shell Section : {shell_section}")

        shell_section.strip()

        lines = shell_section.split('\n')
        ret_value = CommandResult("", "")
        
        if len(lines) == 1:
            # Single-line command, execute directly
            command = lines[0]

            ret_value = self.run_command(command)
            logging.error(f"Execute Shell Directory Line Strip: {ret_value}")

            if ret_value.err == "":
                lines = ret_value.out.splitlines()
                
                if lines:
                    logging.error(f"Execute Shell Directory line: {lines[-1]}")
                    new_dir = lines[-1]  # Assuming the last line of output contains the new working directory
                    logging.error(f"Execute Shell Directory new_dir: {new_dir}")
                    if os.path.isdir(new_dir):
                        os.chdir(new_dir)  # Change to the new working directory in your parent process
                    else:
                        logging.error(f"Invalid directory: {new_dir}")
                else:
                    logging.error("No output to determine the new working directory")
                logging.info(f"Command Return : {ret_value}")
            else:
                logging.info(f"Command Return Error : {ret_value.err} with Command : {command}")
        else:
            # Multi-line command, create a batch file
            batch_filename = 'temp.bat'
            if lines[0].startswith('REM '):
                # Use commented out filename
                batch_filename = lines[0][4:].strip()
                # lines = lines[1:]  # Remove the filename line

            logging.info(f"batch_filename : {batch_filename}")
            with open(batch_filename, 'w') as batch_file:
                batch_file.write('\n'.join(lines))
            print("==== Batch File ====")
            print('\n'.join(lines))
            print("====================")
            user_confirmation = input("Are you sure you want to run this batch file? (yes/no): ")
            logging.info(f"user_confirmation : {user_confirmation}")
            if user_confirmation.lower() != 'yes':
                return "Canceled Command"
            ret_value = self.run_command(batch_filename)
            
            logging.info(f"command output : out: {ret_value.out}, err: {ret_value.err}")
            if batch_filename == 'temp.bat':
                os.remove(batch_filename)  # Remove temporary batch file
                logging.info(f"removing : {batch_filename}")

        return ret_value
    
    def create_process_group(self):
        # Create a new process group
        process_group_id = os.set_handle_inheritance(0, 1)
        return process_group_id

    def run_command(self, command):
        command += " && cd"
        logging.info(f"run command : {command}")

        # Determine the operating system
        is_windows = platform.system() == 'Windows'

        # Set up the subprocess arguments based on the operating system
        popen_args = {
            'args': command,
            'shell': True,
            'text': True,
            'stdin': subprocess.PIPE,
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
        }
        if not is_windows:
            popen_args['preexec_fn'] = os.setsid  # This is UNIX-only
        else:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            popen_args['startupinfo'] = startupinfo
            popen_args['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

        process = subprocess.Popen(**popen_args)
        
        # Wait for the process to finish and get the final output and error
        stdout, stderr = process.communicate()
        
        if stderr == "":
            lines = stdout.strip().split("\n")
            print(f"stdout = {stdout}")
            print(f"stdout = {len(lines)}")
            if lines:
                new_dir = lines[-1]  # Assuming the last line of output contains the new working directory
                print(f"new_dir = {new_dir}")
                if os.path.isdir(new_dir):
                    os.chdir(new_dir)  # Change to the new working directory in your parent process
                else:
                    logging.error(f"Invalid directory: {new_dir}")
            else:
                logging.error("No output to determine the new working directory")

        logging.info(f"run return : out: {stdout}, err: {stderr}")
        
        ret_val = CommandResult(stdout, stderr)
        return ret_val
        
    def handle_input(self, process):
        try:
            while True:
                user_input = input("Enter 'pause' to pause, 'resume' to resume, or 'ctrl+c' to terminate: ").lower()
                # Determine the operating system
                is_windows = platform.system() == 'Windows'

                if not is_windows:
                    # UNIX-like behavior
                    if user_input == 'pause':
                        os.killpg(os.getpgid(process.pid), signal.SIGSTOP)
                    elif user_input == 'resume':
                        os.killpg(os.getpgid(process.pid), signal.SIGCONT)
                    elif user_input == 'ctrl+c':
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    # Windows behavior
                    if user_input == 'ctrl+c':
                        os.kill(process.pid, signal.CTRL_BREAK_EVENT)  # Send CTRL+BREAK to the process
                    else:
                        print("Pausing and resuming are not supported on Windows.")

        except Exception as e:
            logging.error(f"Error in handle_input: {e}")

    def translate_to_command(self, user_input):
        commands = map_possible_commands()
        token_count = get_token_count(commands)
        if token_count > self.llm_len:
            commands = trim_to_token_count(commands, self.llm_len)
            
        logging.info(f"Translate to Command : {user_input}")
        send_prompt = self.settings['command_prompt']
        kwargs = {
             'get_os_name': get_os_name(),
             'commands': commands
        }
        send_prompt = replace_placeholders(send_prompt, **kwargs)
        logging.info(f"Translate use Command : {send_prompt}")
        command_output = self.llm.ask(send_prompt, user_input, model_type=self.settings.get('model_type', 'OpenAI'))
        logging.info(f"Translate return Response : {command_output}")

        if command_output == None:
            command_output = "None"
        if '```shell' in command_output:
            tran_command = self.extract_shell_command(command_output)
            command_output = self.execute_shell_section(tran_command).out
            logging.info(f"Translate Shell Execute : {command_output}")
        elif '```batch' in command_output:
            tran_command = self.extract_batch_command(command_output)
            command_output = self.execute_shell_section(tran_command).out
            logging.info(f"Translate Shell Execute : {command_output}")
        elif '```bash' in command_output:
            tran_command = self.extract_bash_command(command_output)
            command_output = self.execute_shell_section(tran_command).out
            logging.info(f"Translate Shell Execute : {command_output}")
        elif '```python' in command_output:
            tran_command = self.extract_python_script(command_output)
            command_output = self.execute_python_script(tran_command)
            logging.info(f"Translate Python Execute : {command_output}")
        else:
            success, command_output = self.execute_command(command_output)
            command_output = command_output
            logging.info(f"Translate Command Execute : {command_output}")
        

        logging.info(f"Translate command output : {command_output}")

        return command_output

    def execute_command(self, command):
        try:
            logging.info(f"Execute Command : {command}")
            result = self.run_command(command)
            if result.err:
                logging.info(f"Execute Error : {result.err}")
                return False, f"Command : {command}, Error: {result.err}"
            
            logging.info(f"Execute Output : {result.out}")

            out_text = result.out
            if out_text == "":
                out_text = f"No output for '{command}'"

            return True, out_text
        except Exception as e:
            return False, str(e)

    def translate_output(self, output):
        logging.info(f"Translate Output : {output}")
        token_count = get_token_count(output)
        if token_count > self.llm_output_size:
            output = trim_to_token_count(output, self.llm_output_size)

        send_prompt = self.settings['display_prompt']
        kwargs = {
             'get_os_name': get_os_name()
        }
        send_prompt = replace_placeholders(send_prompt, **kwargs)
        logging.info(f"Translate Output Display Prompt : {send_prompt}")
        display_output = self.llm.ask(send_prompt, output, model_type=self.settings.get('model_type', ModelTypes('OpenAI')))
        logging.info(f"Translate Output Display Response : {display_output}")
        return display_output

    def display_output(self, output):
        logging.info(f"Display Output : {output}")
        print_colored_text(output)

    def display_about(self):
        print_colored_text("[bold][yellow]ShellSpeak\n======================================================\n[white]AI powered Console Input\nVisit: https://github.com/TheCompAce/ShellSpeak\nDonate: @BradfordBrooks79 on Venmo\n\n[grey]Tip: Type 'help' for Help.\n[yellow]======================================================\n")

    def display_help(self):
        print_colored_text("[bold][yellow]ShellSpeak Help\n======================================================\n[white]Type:\n'exit' to close ShellSpeak\n'user: /command/' pass a raw command to execute then reply threw the AI\n'about' Shows the About Information\n'help' Shows this Help information.\n[yellow]======================================================\n")

    def run(self):
        self.display_about()
        while True:
            self.settings = load_settings(self.settingsRoot)

            user_input = self.capture_input()
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'about':
                self.display_about()
            elif user_input.lower() == 'help':
                self.display_help()
            else:
                if user_input.lower().startswith('user: '):
                    # Bypass AI translation and send raw command to the OS
                    raw_command = user_input[6:]  # Extract the command part from user_input
                    result = self.run_command(raw_command)
                    translated_output = self.translate_output(result.out)
                    self.display_output(f"Output:\n{result.out}\nError:\n{result.err}")
                else:
                    # Continue with AI translation for the command
                    translated_command = self.translate_to_command(user_input)
                    translated_output = self.translate_output(translated_command)
                    self.display_output(translated_output)
