# Import necessary modules
import asyncio
import datetime
import json
import os
import platform
import queue
import re
import subprocess
import logging
import signal
import base64
import threading
import spacy
from pygments import lexers
from modules.command_result import CommandResult

from modules.llm import LLM, ModelTypes
from modules.run_command import CommandRunner
from modules.utils import get_file_size, get_token_count, load_settings, map_possible_commands, get_os_name, print_colored_text, capture_styled_input, read_file, trim_to_right_token_count, trim_to_token_count, replace_placeholders
from modules.vectors import find_relevant_file_segments


nlp = spacy.load("en_core_web_sm")
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class ShellSpeak:
    def __init__(self, settings, base_path):
        self.llm_len = int(settings.get("llm_size", 4097))
        self.llm_history_len = int(settings.get("llm_history_size", 2000))
        self.llm_file_len = int(settings.get("llm_file_size", 2000))

        self.temp_file = settings.get("temp_file", "temp")

        self.llm_output_size = int(settings.get("llm_output_size", 4097))
        self.use_cache = settings.get("use_cache", False)
        self.cache_file = settings.get("cache_file", None)

        self.vector_for_commands = settings.get("vector_for_commands", False)
        self.vector_for_history = settings.get("vector_for_history", True)

        

        self.settings = settings
        self.command_history = ""
        self.settingsRoot = base_path

        self.files = []

        self.llm = LLM(model_type=ModelTypes(self.settings.get('model', "OpenAI")), use_cache=self.use_cache, cache_file=self.cache_file) #Zephyr7bBeta

        self.command_runner = CommandRunner(self)

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

    def detect_language(self, code):
        try:
            lexer = lexers.guess_lexer(code)
            return lexer.name
        except lexers.ClassNotFound:
            return None
    
    async def execute_python_script(self, python_section):
        lines = python_section.split('\n')
        if len(lines) == 1:
            # Single-line script, execute directly
            script = lines[0]
            # script = f"{self.settings['python_command_prompt']}\n{script}"
            output = await self.run_python_script(script)
            return output
        else:
            # Multi-line script, create a python file
            python_filename = f'{self.temp_file}.py'
            if lines[0].startswith('#'):
                # Use commented out filename
                python_filename = lines[0][1:].strip()
                lines = lines[1:]  # Remove the filename line

            script = '\n'.join(lines)
            script = f"{self.settings['python_command_prompt']}\n{script}"

            with open(python_filename, 'w') as python_file:
                python_file.write(script)
            self.show_file("Python File", script.split('\n'))
            user_confirmation = input("Are you sure you want to run this Python script? (yes/no): ")
            if user_confirmation.lower() != 'yes':
                return CommandResult("", "Run python file Canceled.")
            output = await self.run_python_script(python_filename)
            if python_filename == f'{self.temp_file}.py':
                os.remove(python_filename)  # Remove temporary python file
            return output
    
    async def run_python_script(self, script):
        # If the script is a file, use 'python filename.py' to execute
        if script.endswith('.py'):
            command = f'python -u {script}'
        else:
            command = f'python -u -c "{script}"'
        result = await self.run_command(command)
        return CommandResult(result.out, result.err)
    
    def extract_script_command(self, script_type, text):
        match = re.search(rf'```{script_type}(.*?)```', text, re.DOTALL)
        if match:
            shell_section = match.group(1).strip()
        else:
            logging.error(f"No {script_type} section found")
            shell_section = None

        return shell_section

    
    

    async def execute_shell_section(self, shell_section):

        logging.info(f"Executing Shell Section : {shell_section}")

        shell_section.strip()

        lines = shell_section.split('\n')
        ret_value = CommandResult("", "")
        
        if len(lines) == 1:
            # Single-line command, execute directly
            command = lines[0]

            ret_value = await self.run_command(command)
            logging.error(f"Execute Shell Directory Line Strip: {ret_value}")

        else:
            # Multi-line command, create a batch file
            batch_filename = f'{self.temp_file}.bat'
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
            ret_value = await self.run_command(batch_filename)
            
            logging.info(f"command output : out: {ret_value.out}, err: {ret_value.err}")
            if batch_filename == f'{self.temp_file}.bat':
                os.remove(batch_filename)  # Remove temporary batch file
                logging.info(f"removing : {batch_filename}")

        return ret_value
    
    def create_process_group(self):
        # Create a new process group
        process_group_id = os.set_handle_inheritance(0, 1)
        return process_group_id

    async def run_command(self, command):
        command += " && cd"
        logging.info(f"run command : {command}")

        stdout, stderr = await self.command_runner.run(command)


        if stderr == "":
            lines = stdout.strip().split("\n")
            if lines:
                new_dir = lines[-1]  # Assuming the last line of output contains the new working directory
                if os.path.isdir(new_dir):
                    os.chdir(new_dir)  # Change to the new working directory in your parent process
                    # Remove the last line containing the new directory from the output
                    lines = lines[:-1]
                    stdout = '\n'.join(lines)
                else:
                    logging.error(f"Invalid directory: {new_dir}")
            else:
                logging.error("No output to determine the new working directory")
        else:
            stderr = f"Command : {command}, Error: {stderr}"

        logging.info(f"run return : out: {stdout}, err: {stderr}")

        ret_val = CommandResult(stdout, stderr)
        return ret_val
    
        
    def format_for_display(self, input, output):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.command_history += f"History: [Time: {timestamp}\nInput: {input}\nOutput: {output}]\n"
        self.display_output(output)

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

    def shrink_file_data(self, file_data, target_tokens):
        if len(file_data) > nlp.max_length:
            file_data = file_data[:len(file_data) - nlp.max_length]

        # Get the current token count of file_data
        current_tokens = get_token_count(file_data)

        if current_tokens > target_tokens:
            # Estimate the number of characters to keep based on the average token length
            average_token_length = len(file_data) / current_tokens
            chars_to_keep = int(target_tokens * average_token_length)
            
            # Only keep the last part of file_data
            truncated_data = file_data[-chars_to_keep:]
            return truncated_data

        # If the file_data is already within the limit, return it as is
        return file_data


    def find_relevant_data(file_data, target_tokens):
        # Your logic here to find relevant information within the token count
        return file_data[:target_tokens]

    def expand_directories(self, file_paths, exclusions):
        new_file_list = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # If the path is a directory, ask the user whether to include its files
                user_decision = input(f"The path '{file_path}' is a directory. Do you want to add all files in this directory? (y/n): ")
                if user_decision.lower() == 'y':
                    # If yes, walk through the directory and add all files
                    for root, dirs, files in os.walk(file_path):
                        # Remove excluded directories so os.walk doesn't traverse them
                        dirs[:] = [d for d in dirs if d not in exclusions]
                        for name in files:
                            if name not in exclusions:
                                new_file_list.append(os.path.join(root, name))
                else:
                    # If no, inform the user that the directory is being skipped
                    print(f"Skipping directory '{file_path}'.")
            else:
                # If the path is a file, just add it to the list
                if os.path.basename(file_path) not in exclusions:
                    new_file_list.append(file_path)
        return new_file_list



    async def translate_to_command(self, user_input):
        user_command_prompt = self.settings['user_command_prompt']
        

        send_prompt = self.settings['command_prompt']
        max_llm = (self.llm_len - 80) #80 is used to padd json formating of System Messages and over all prompt size.
        max_llm -= get_token_count(send_prompt)
        max_llm -= get_token_count(user_input)
        
        set_command_history = self.command_history
        token_count = get_token_count(set_command_history)
        if token_count > self.llm_history_len:
            if self.vector_for_history:
                relevant_segments = find_relevant_file_segments(
                            history_text= user_input,
                            file_data=set_command_history,
                            window_size=self.llm_history_len, # or any other size you deem appropriate (8124)
                            overlap=100,      # or any other overlap size you deem appropriate
                            top_k=1           # or any other number of segments you deem appropriate
                        )
                set_command_history = '/n.../n'.join(relevant_segments)
            else:
                set_command_history = trim_to_right_token_count(set_command_history, self.llm_history_len)
        history_tokens = get_token_count(set_command_history)
        max_llm -= history_tokens
        logging.info(f"Translate to Command History Token Count : {history_tokens}")


        command_history = json.dumps(set_command_history)

        set_command_files_data = []
        total_tokens = 0

        # Extract file paths and exclusion list from user_input
        file_paths = re.findall(r'file:\s*(".*?"|\S+)', user_input)
        
        # Remove quotes from file paths, if present
        self.files = [fp.strip('"') for fp in file_paths]
        for f, file in enumerate(self.files):
            exclusions = file.split(',')
            file_path = exclusions[0]

            exclusions.pop(0)
            self.files[f] = file_path
            self.exclusions = exclusions
            self.files = self.expand_directories(self.files, self.exclusions)

            # Use the new function to expand directories into file lists
            self.files = self.expand_directories(self.files, self.exclusions)

        if len(self.files) > 0:
            total_size = 0
            total_data = ""
            files_data = []
            
            for file in self.files:
                file_data_content = read_file(file)  # Note: Changed to 'file_data_content'
                if len(file_data_content) > nlp.max_length:
                    file_data_content = file_data_content[:len(file_data_content) - nlp.max_length]

                file_data = {
                    "file": file,
                    "file_data": file_data_content,
                    "file_size": int(get_file_size(file)),
                    "file_tokens": get_token_count(file_data_content)  # Note: Changed to 'file_data_content'
                }
                
                total_size += file_data["file_size"]
                total_data += file_data["file_data"]

                files_data.append(file_data)

            # Sort files_data by file_tokens in descending order
            files_data = sorted(files_data, key=lambda x: x['file_tokens'], reverse=True)

            remaining_tokens = self.llm_file_len
            remaining_tokens_split = int(remaining_tokens / len(files_data)) + 1
            new_files_data = []
            for f, file in enumerate(files_data):
                if file["file_tokens"] > remaining_tokens_split:
                    file["fileIndex"] = f
                    file["file_tokens"] = remaining_tokens_split
                    new_files_data.append(file)
                else:
                    remaining_tokens -= file["file_tokens"]
                    div_val = (len(files_data) - (len(files_data) -  len(new_files_data)))
                    if div_val == 0:
                        div_val = 1

                    remaining_tokens_split = int(remaining_tokens / div_val)
                    
            # remaining_tokens = self.llm_file_len
            # remaining_tokens_split = int(remaining_tokens / len(files_data)) + 1

            if len(new_files_data) > 0:
                for new_file in new_files_data:
                    print(f"File {new_file['file']} Trimming")
                    relevant_segments = find_relevant_file_segments(
                            history_text=command_history + "\n"+ user_input,
                            file_data=new_file['file_data'],
                            window_size=new_file['file_tokens'], # or any other size you deem appropriate (8124)
                            overlap=100,      # or any other overlap size you deem appropriate
                            top_k=1           # or any other number of segments you deem appropriate
                        )
                    new_file['file_data'] = '/n.../n'.join(relevant_segments)
                    new_file['file_tokens'] = get_token_count(new_file['file_data'])

                    files_data[new_file["fileIndex"]] = new_file

            total_tokens = 0
            for file_data in files_data:
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
        logging.info(f"Translate to Command File Token Count : {total_tokens}")

        max_llm -= total_tokens

        commands = map_possible_commands()
        token_count = get_token_count(commands)
        if token_count > max_llm:
            if self.vector_for_commands:
                relevant_segments = find_relevant_file_segments(
                            history_text=command_history + "\n"+ user_input,
                            file_data=commands,
                            window_size=max_llm, # or any other size you deem appropriate (8124)
                            overlap=100,      # or any other overlap size you deem appropriate
                            top_k=1           # or any other number of segments you deem appropriate
                        )
                commands = '/n.../n'.join(relevant_segments)
                commands = commands.replace(' .exe', '.exe').replace(' .bat', '.bat').replace(' .com', '.com').replace(' .sh', '.sh')
            else:
                commands = trim_to_token_count(commands, max_llm)
        
        command_tokens = get_token_count(commands)
        logging.info(f"Translate to Command File Token Count : {command_tokens}")
        
        logging.info(f"Translate to Command : {user_input}")

        kwargs = {
            'user_prompt': user_input,
            'get_os_name': get_os_name(),
            'commands': commands,
            'command_history': command_history,
            'command_files_data': command_files_data
        }
        user_command_prompt = replace_placeholders(user_command_prompt, **kwargs)
        system_command_prompt = replace_placeholders(send_prompt, **kwargs)

        user_tokens = get_token_count(user_command_prompt)
        system_tokens = get_token_count(system_command_prompt)
        logging.info(f"Translate to Command User Token Count : {user_tokens}")
        logging.info(f"Translate to Command System Token Count : {system_tokens}")

        logging.info(f"Translate use System Prompt : {system_command_prompt}")
        logging.info(f"Translate use User Prompt : {user_command_prompt}")
        # command_output = self.llm.ask(system_command_prompt, user_command_prompt, model_type=ModelTypes(self.settings.get('model', "OpenAI")))
        # loop = asyncio.get_event_loop()
        # command_output = await loop.run_in_executor(None, lambda: self.llm.ask(system_command_prompt, user_command_prompt, model_type=ModelTypes(self.settings.get('model', "OpenAI"))))
        command_output = await self.llm.async_ask(system_command_prompt, user_command_prompt, model_type=ModelTypes(self.settings.get('model', "OpenAI")))
        logging.info(f"Translate return Response : {command_output}")

        if command_output == None:
            command_output = "Error with Command AI sub system!"
        elif len(command_output) > 9 and command_output[:9] == "RESPONSE:":
            print(f"command_output = {command_output}")
            command_output = command_output[9:].strip()
        elif len(command_output) > 8 and command_output[:8] == "COMMAND:":
            command_output = command_output[8:].strip()
        
            success, command_output = await self.execute_command(command_output)
            if not success:
                print(f"Exe Error: {command_output.err}")
                command_output = command_output.err
            else:
                command_output = command_output.out
            logging.info(f"Translate Command Execute : {command_output}")
        else:
            check_list = ["shell", "batch", "bash", "python"]
            multi_scripts = []
            for check in check_list:
                # Create a regex pattern based on the current check
                pattern = re.compile(f'```{check}(.*?)```', re.DOTALL)
                
                # Find all occurrences of the pattern in command_output
                for match in pattern.finditer(command_output):
                    # Extract the matched script content
                    script_content = match.group(1)
                    
                    # Assuming self.check_script processes the script content
                    #check_command_output = self.check_script(check, script_content)
                    check_command_output = script_content
                    
                    add_match = {
                        "check": check,
                        "check_command": check_command_output
                    }
                    
                    multi_scripts.append(add_match)

            canceled = False
            if len(multi_scripts) > 0:
                if len(multi_scripts) > 1:
                    while True:
                        print_colored_text(f"[yellow]===== Base Command =====\n[white]{command_output}\n[yellow]========================")
                        print_colored_text(f"[yellow]Found {len(multi_scripts)} diffrent scripts in the response")
                        for m, multi_script in enumerate(multi_scripts):
                            print(f"{m + 1} : {multi_script['check']}")
                        
                        print(f"{len(multi_scripts) + 1} : Cancel")
                        get_check = capture_styled_input("[yellow]Select the type of script you want to run: ")

                        if get_check.isdigit:
                            get_check = int(get_check)
                            if get_check != len(multi_scripts) + 1 and get_check > 0 and get_check <= len(multi_scripts):
                                command_output = multi_scripts[get_check - 1]["check_command"]
                                break
                            elif get_check == len(multi_scripts) + 1:
                                canceled = True
                                break
                else:
                    command_output = multi_scripts[0]["check_command"]

            else:
                code_type = self.detect_language(command_output)
                if code_type == "Python":
                   check = "python"
                else:
                   print(f"code_type = {code_type}")
                   canceled = True
        
            if not canceled:
                if check == "shell" or check == "batch" or check == "bash":
                    command_output = await self.execute_shell_section(command_output)
                else:
                    command_output = await self.execute_python_script(command_output)

                if command_output.err != "":
                    print(f"Shell Error: {command_output.out}")
                    command_output = command_output.err
                else:    
                    command_output = command_output.out

                logging.info(f"Translate Shell Execute : {command_output}")

        logging.info(f"Translate command output : {command_output}")

        return command_output
    
    def check_script(self, code_type, text):
        command_output = text
        if f'```{code_type}' in text:
            command_output = self.extract_script_command(code_type, text)
            logging.info(f"Translate '{code_type}' Code : {text}")

        return command_output

    async def execute_command(self, command):
        try:
            logging.info(f"Execute Command : {command}")
            result = await self.run_command(command)
            if result.err:
                logging.info(f"Execute Error : {result.err}")
                return False, result
            
            logging.info(f"Execute Output : {result.out}")

            return True, result
        except Exception as e:
            return False, CommandResult("", str(e))

    def translate_output(self, output, is_internal=False):
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
             'command_history': set_command_history,
             'internal_script': str(is_internal)
        }
        send_prompt = replace_placeholders(send_prompt, **kwargs)

        logging.info(f"Translate Output Display System Prompt : {send_prompt}")
        logging.info(f"Translate Output Display User Prompt : {output}")
        display_output = self.llm.ask(send_prompt, output, model_type=ModelTypes(self.settings.get('model', "OpenAI")))
        logging.info(f"Translate Output Display Response : {display_output}")
        return display_output

    def display_output(self, output):
        logging.info(f"Display Output : {output}")
        print_colored_text(output)

    def display_about(self):
        print_colored_text("[bold][yellow]======================================================\nShellSpeak\n======================================================\n[white]AI powered Console Input\nVisit: https://github.com/TheCompAce/ShellSpeak\nDonate: @BradfordBrooks79 on Venmo\n\n[grey]Type 'help' for Help.\n[yellow]======================================================\n")

    def display_help(self):
        print_colored_text("[bold][yellow]======================================================\nShellSpeak Help\n======================================================\n[white]Type:\n'exit': to close ShellSpeak\n'user: /command/': pass a raw command to execute then reply threw the AI\n'file: /filepath/': adds file data to the command prompt. (use can send a folder path, using ',' to exclude folders and files.)\n'clm': Clear command Memory\n'rset': Reloads the settings file (this happens on every loading of the prompt.)\n'about': Shows the About Information\n'help': Shows this Help information.\n[yellow]======================================================\n")

    async def run(self):
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
            elif user_input.lower() == 'rset':
                print("Settings Updated")
            elif user_input.lower() == 'clm':
                self.command_history = ""
                # self.command_history += f"Command Input: {user_input}\nCommand Output: Command History cleared.\n"
                self.display_output(f"Command Memory (History) Cleared.")
            else:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if user_input.lower().startswith('user: '):
                    # Bypass AI translation and send raw command to the OS
                    raw_command = user_input[6:]  # Extract the command part from user_input
                    try:
                        result = await self.run_command(raw_command)
                    except Exception as e:
                        translated_command = e
                    translated_output = self.translate_output(result.out)
                    self.command_history += f"History: [Time: {timestamp}\nInput: {user_input}\nOutput: {result.out} Error: {result.err}]\n"
                    # self.display_output(f"Output:\n{result.out}\nError:\n{result.err}")
                    self.display_output(translated_output)
                else:
                    # Continue with AI translation for the command
                    try:
                       translated_command = await self.translate_to_command(user_input)
                    except Exception as e:
                        translated_command = e
                    # if translated_command.err == "":
                    #    translated_output = self.translate_output(translated_command)
                    #    self.command_history += f"Command Input: {user_input}\nCommand Output: {translated_output}\n"
                    #    self.display_output(translated_output)
                    #else:
                    
                    self.command_history += f"History: [Time: {timestamp}\nInput: {user_input}\nOutput: {translated_command}]\n"
                    translated_output = self.translate_output(translated_command)
                    self.display_output(translated_output)
