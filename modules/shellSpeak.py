# Import necessary modules
import json
import os
import platform
import re
import subprocess
import logging
import signal
import base64

from modules.llm import LLM, ModelTypes
from modules.utils import get_file_size, get_token_count, load_settings, map_possible_commands, get_os_name, print_colored_text, capture_styled_input, read_file, trim_to_right_token_count, trim_to_token_count, replace_placeholders
from modules.vectors import find_relevant_file_segments

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CommandResult:
    def __init__(self, stdout, stderr):
        self.out = stdout
        self.err = stderr

class ShellSpeak:
    def __init__(self, settings, base_path):

        self.llm_len = int(settings.get("llm_size", 4097))
        self.llm_history_len = int(settings.get("llm_history_size", 2000))
        self.llm_file_len = int(settings.get("llm_file_size", 2000))

        self.llm_output_size = int(settings.get("llm_output_size", 4097))
        self.use_cache = settings.get("use_cache", False)
        self.cache_file = settings.get("cache_file", None)
        self.settings = settings
        self.command_history = ""
        self.settingsRoot = base_path

        self.files = []

        self.llm = LLM(model_type=ModelTypes(self.settings.get('model', "OpenAI")), use_cache=self.use_cache, cache_file=self.cache_file) #Zephyr7bBeta

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
    
    def show_file(self, caption, body):
        print_colored_text(f"[yellow]==== {caption} ====")
        print_colored_text('[cyan]' + '\n'.join(body))
        print_colored_text("[yellow]====================")
    
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
            self.show_file("Python File", lines)
            user_confirmation = input("Are you sure you want to run this Python script? (yes/no): ")
            if user_confirmation.lower() != 'yes':
                return CommandResult("", "Run python file Canceled.")
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
    
    def extract_plain_text(self, text):
        match = re.search(r'```plaintext(.*?)```', text, re.DOTALL)
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
            self.show_file("Batch File", lines)
            user_confirmation = input("Are you sure you want to run this batch file? (yes/no): ")
            logging.info(f"user_confirmation : {user_confirmation}")
            if user_confirmation.lower() != 'yes':
                return CommandResult("", "Run batch file Canceled.")
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
            if lines:
                new_dir = lines[-1]  # Assuming the last line of output contains the new working directory
                if os.path.isdir(new_dir):
                    os.chdir(new_dir)  # Change to the new working directory in your parent process
                else:
                    logging.error(f"Invalid directory: {new_dir}")
            else:
                logging.error("No output to determine the new working directory")
        else:
            stderr = f"Command : {command}, Error: {stderr}"

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

    def shrink_file_data(file_data, target_tokens):
        # Your logic here to shrink the file data to target_tokens
        # For now, we'll just truncate the data
        truncated_data = file_data[:target_tokens]
        return truncated_data

    def find_relevant_data(file_data, target_tokens):
        # Your logic here to find relevant information within the token count
        return file_data[:target_tokens]

    def expand_directories(self, file_paths):
        new_file_list = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # If the path is a directory, ask the user whether to include its files
                user_decision = input(f"The path '{file_path}' is a directory. Do you want to add all files in this directory? (y/n): ")
                if user_decision.lower() == 'y':
                    # If yes, walk through the directory and add all files
                    for root, dirs, files in os.walk(file_path):
                        for name in files:
                            new_file_list.append(os.path.join(root, name))
                else:
                    # If no, inform the user that the directory is being skipped
                    print(f"Skipping directory '{file_path}'.")
            else:
                # If the path is a file, just add it to the list
                new_file_list.append(file_path)
        return new_file_list


    def translate_to_command(self, user_input):
        send_prompt = self.settings['command_prompt']
        max_llm = (self.llm_len - 80) #80 is used to padd json formating of System Messages and over all prompt size.
        max_llm -= get_token_count(send_prompt)
        max_llm -= get_token_count(user_input)
        
        set_command_history = self.command_history
        token_count = get_token_count(set_command_history)
        if token_count > self.llm_history_len:
            set_command_history = trim_to_right_token_count(set_command_history, self.llm_history_len)

        ext_tokens = get_token_count(set_command_history)

        command_history = json.dumps(set_command_history)

        set_command_files_data = []
        total_tokens = 0

        # Extract file paths from user_input
        file_paths = re.findall(r'file:\s*(".*?"|\S+)', user_input)
        
        # Remove quotes from file paths, if present
        self.files = [fp.strip('"') for fp in file_paths]

        # Use the new function to expand directories into file lists
        self.files = self.expand_directories(self.files)

        if len(self.files) > 0:
            total_size = 0
            total_data = ""
            files_data = []
            
            for file in self.files:
                file_data_content = read_file(file)  # Note: Changed to 'file_data_content'
                file_data = {
                    "file": file,
                    "file_data": file_data_content,
                    "file_size": int(get_file_size(file)),
                    "file_tokens": get_token_count(file_data_content)  # Note: Changed to 'file_data_content'
                }
                
                total_size += file_data["file_size"]
                total_data += file_data["file_data"]

                files_data.append(file_data)

            total_tokens = get_token_count(total_data)

            # Sort files_data by file_tokens in descending order
            files_data = sorted(files_data, key=lambda x: x['file_tokens'], reverse=True)

            remaining_tokens = self.llm_file_len
            new_files_data = []
            sized_fixed = False
            if total_tokens > self.llm_file_len:
                print(f"File Trimming")
                # Step 1: Reduce size of the largest files first
                for file_data in files_data:
                    if remaining_tokens <= 0:
                        break
                    
                    if file_data['file_tokens'] > remaining_tokens:
                        relevant_segments = find_relevant_file_segments(
                            history_text=command_history,
                            file_data=file_data['file_data'],
                            window_size=8192, # or any other size you deem appropriate
                            overlap=100,      # or any other overlap size you deem appropriate
                            top_k=5           # or any other number of segments you deem appropriate
                        )
                        
                        file_data['file_data'] = '/n.../n'.join(relevant_segments)  # Join the segments into a single string
                        file_data['file_tokens'] = get_token_count(file_data['file_data'])
                        file_data['adjusted_tokens'] = file_data['file_tokens']
                    else:
                        file_data['adjusted_tokens'] = file_data['file_tokens']
                    
                    remaining_tokens -= file_data['adjusted_tokens']
                    new_files_data.append(file_data)

                    sized_fixed = True

                # Step 2: Reclaim size for smaller files, if possible
                for file_data in reversed(new_files_data):  # Start from smallest files
                    if remaining_tokens <= 0:
                        break

                    extra_tokens = min(file_data['file_tokens'] - file_data['adjusted_tokens'], remaining_tokens)
                    new_files_data['adjusted_tokens'] += extra_tokens
                    remaining_tokens -= extra_tokens

                files_data = new_files_data

            total_tokens = 0
            for file_data in files_data:
                if sized_fixed:
                    total_tokens += file_data["adjusted_tokens"]
                else:
                    total_tokens += file_data["file_tokens"]

                 # Check if the file_data is binary and encode it with base64 if so
                try:
                    # This will work if 'file_data' is text
                    encoded_data = json.dumps(file_data['file_data'])
                except TypeError:
                    # If 'file_data' is binary, encode it with base64
                    encoded_data = base64.b64encode(file_data['file_data']).decode('utf-8')

                add_command_files_data = {
                    "file:": file_data["file"],
                    "data:": encoded_data
                }

                set_command_files_data.append(add_command_files_data)
        

        command_files_data = json.dumps(set_command_files_data)

        ext_tokens += total_tokens


        commands = map_possible_commands()
        token_count = get_token_count(commands)
        llm_left = (max_llm - ext_tokens)
        if token_count > llm_left:
            commands = trim_to_token_count(commands, llm_left)
        
            
        logging.info(f"Translate to Command : {user_input}")
        

        kwargs = {
             'get_os_name': get_os_name(),
             'commands': commands,
             'command_history': command_history,
             'command_files_data': command_files_data
        }
        send_prompt = replace_placeholders(send_prompt, **kwargs)
        logging.info(f"Translate use Command : {send_prompt}")
        command_output = self.llm.ask(send_prompt, user_input, model_type=ModelTypes(self.settings.get('model', "OpenAI")))
        logging.info(f"Translate return Response : {command_output}")

        if command_output == None:
            command_output = "Error with Command AI sub system!"
        elif len(command_output) > 5 and command_output[:5] == "TALK:":
            command_output = command_output[5:]
        elif '```shell' in command_output:
            tran_command = self.extract_shell_command(command_output)
            command_output = self.execute_shell_section(tran_command)
            if command_output.err != "":
                print(f"Shell Error: {command_output.out}")
                command_output = command_output.err
            else:    
                command_output = command_output.out
            logging.info(f"Translate Shell Execute : {command_output}")
        elif '```batch' in command_output:
            tran_command = self.extract_batch_command(command_output)
            command_output = self.execute_shell_section(tran_command)
            if command_output.err != "":
                print(f"Batch Error: {command_output.out}")
                command_output = command_output.err
            else:
                command_output = command_output.out
            logging.info(f"Translate Shell Execute : {command_output}")
        elif '```bash' in command_output:
            tran_command = self.extract_bash_command(command_output)
            command_output = self.execute_shell_section(tran_command)
            if command_output.err != "":
                print(f"Bash Error: {command_output.out}")
                command_output = command_output.err
            else:
                command_output = command_output.out
            logging.info(f"Translate Shell Execute : {command_output}")
        elif '```python' in command_output:
            tran_command = self.extract_python_command(command_output)
            command_output = self.execute_python_script(tran_command)
            logging.info(f"Translate Python Execute : {command_output}")
        elif '```plaintext' in command_output:
            command_output = self.extract_plain_text(command_output)            
        else:
            success, command_output = self.execute_command(command_output)
            if not success:
                print(f"Exe Error: {command_output.err}")
                command_output = command_output.err
            else:
                command_output = command_output.out
            logging.info(f"Translate Command Execute : {command_output}")
        

        logging.info(f"Translate command output : {command_output}")

        return command_output

    def execute_command(self, command):
        try:
            logging.info(f"Execute Command : {command}")
            result = self.run_command(command)
            if result.err:
                logging.info(f"Execute Error : {result.err}")
                return False, result
            
            logging.info(f"Execute Output : {result.out}")

            return True, result
        except Exception as e:
            return False, CommandResult("", str(e))

    def translate_output(self, output):
        logging.info(f"Translate Output : {output}")
        send_prompt = self.settings['display_prompt']
        total_tokens = self.llm_output_size - (get_token_count(send_prompt) + get_token_count(output) + 80)

        set_command_history = self.command_history
        token_count = get_token_count(set_command_history)

        if token_count > total_tokens:
            set_command_history = trim_to_right_token_count(set_command_history, total_tokens)

        # ext_tokens = token_count

        # token_count = get_token_count(output)
        # if token_count > self.llm_output_size - ext_tokens:
        #    output = trim_to_token_count(output, self.llm_output_size - ext_tokens)

        kwargs = {
             'get_os_name': get_os_name(),
             'command_history': set_command_history
        }
        send_prompt = replace_placeholders(send_prompt, **kwargs)

        logging.info(f"Translate Output Display Prompt : {send_prompt}")
        display_output = self.llm.ask(send_prompt, output, model_type=ModelTypes(self.settings.get('model', "OpenAI")))
        logging.info(f"Translate Output Display Response : {display_output}")
        return display_output

    def display_output(self, output):
        logging.info(f"Display Output : {output}")
        print_colored_text(output)

    def display_about(self):
        print_colored_text("[bold][yellow]ShellSpeak\n======================================================\n[white]AI powered Console Input\nVisit: https://github.com/TheCompAce/ShellSpeak\nDonate: @BradfordBrooks79 on Venmo\n\n[grey]Tip: Type 'help' for Help.\n[yellow]======================================================\n")

    def display_help(self):
        print_colored_text("[bold][yellow]ShellSpeak Help\n======================================================\n[white]Type:\n'exit': to close ShellSpeak\n'user: /command/': pass a raw command to execute then reply threw the AI\n'file: /filepath/': adds file data to the command prompt.\n'clm': Clear command Memory\n'about': Shows the About Information\n'help': Shows this Help information.\n[yellow]======================================================\n")

    def run(self):
        self.display_about()
        while True:
            self.settings = load_settings(self.settingsRoot)
            self.files = []

            user_input = self.capture_input()
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'about':
                self.display_about()
            elif user_input.lower() == 'help':
                self.display_help()
            elif user_input.lower() == 'clm':
                self.command_history = ""
                # self.command_history += f"Command Input: {user_input}\nCommand Output: Command History cleared.\n"
                self.display_output(f"Command Memory cleared")
            else:
                if user_input.lower().startswith('user: '):
                    # Bypass AI translation and send raw command to the OS
                    raw_command = user_input[6:]  # Extract the command part from user_input
                    result = self.run_command(raw_command)
                    translated_output = self.translate_output(result.out)
                    self.command_history += f"History: [Command Input: {user_input}\nCommand Output: {result.out} Command Error: {result.err}]\n"
                    # self.display_output(f"Output:\n{result.out}\nError:\n{result.err}")
                    self.display_output(translated_output)
                else:
                    # Continue with AI translation for the command
                    translated_command = self.translate_to_command(user_input)
                    # if translated_command.err == "":
                    #    translated_output = self.translate_output(translated_command)
                    #    self.command_history += f"Command Input: {user_input}\nCommand Output: {translated_output}\n"
                    #    self.display_output(translated_output)
                    #else:
                    self.command_history += f"History: [Command Input: {user_input}\nCommand Output: {translated_command}]\n"
                    self.display_output(translated_command)
