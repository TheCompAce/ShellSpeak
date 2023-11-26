import json
import os
import platform
import unicodedata
import spacy
import re
from rich.console import Console

nlp = spacy.load("en_core_web_sm")

console = Console()

def get_os_name():
    return platform.system()

def map_possible_commands():
    # Get the operating system name
    os_name = platform.system().lower()
    
    # Get the PATH environment variable
    path_variable = os.environ.get('PATH', '')
    
    # Split it into individual directories
    directories = path_variable.split(os.pathsep)
    
    # Initialize a set to store unique command names
    unique_commands = set()
    
    # List of wanted file extensions for Windows
    windows_wanted_extensions = ['.exe', '.bat', '.com', '.sh']
    
    for directory in directories:
        try:
            # List all files in the directory
            files = os.listdir(directory)
            
            # Filter out executable files and add them to the set
            for file in files:
                file_path = os.path.join(directory, file)
                
                # Get the file extension
                _, extension = os.path.splitext(file)
                
                if os.access(file_path, os.X_OK):
                    if os_name == 'windows':
                        if extension.lower() in windows_wanted_extensions:
                            file = file.replace(f'{windows_wanted_extensions}', '')
                            unique_commands.add(file)
                    else:
                        # On Unix-like systems, rely on executable permission
                        unique_commands.add(file)
                    
        except FileNotFoundError:
            # Directory in PATH does not exist, skip
            continue
        except PermissionError:
            # Don't have permission to access directory, skip
            continue
    
    commands_str = ','.join(unique_commands)
    return commands_str

def print_colored_text(text, end_newline=True):
    try:
        end = "\n" if end_newline else ""
        console.print(text, end=end)
    except Exception as e:
        print(text)

def capture_styled_input(prompt):
    # Print the prompt without a newline at the end
    print_colored_text(prompt, end_newline=False)
    
    # Capture and return user input
    return input()

# Load settings from a JSON file
def load_settings(filepath):
    try:
        with open(os.path.join(filepath, "settings.json"), 'r') as f:
            settings = json.load(f)
            chk_file = os.path.join(filepath, settings['command_prompt'])
            if os.path.isfile(chk_file):
                with open(chk_file, 'r') as f:
                    settings['command_prompt'] = f.read()
            
            chk_file = os.path.join(filepath, settings['display_prompt'])
            if os.path.isfile(chk_file):
                with open(chk_file, 'r') as f:
                    settings['display_prompt'] = f.read()

            chk_file = os.path.join(filepath, settings['user_command_prompt'])
            if os.path.isfile(chk_file):
                with open(chk_file, 'r') as f:
                    settings['user_command_prompt'] = f.read()

            chk_file = os.path.join(filepath, settings['python_command_prompt'])
            if os.path.isfile(chk_file):
                with open(chk_file, 'r') as f:
                    settings['python_command_prompt'] = f.read()

        return settings
    except FileNotFoundError:
        return {}


def replace_placeholders(text, **kwargs):
    """
    Replaces placeholders in the given text with the values provided.

    Parameters:
    - text (str): The text containing placeholders.
    - **kwargs: The values to replace the placeholders with.

    Returns:
    - str: The text with placeholders replaced.
    """

    # Define a regular expression pattern to match placeholders like {placeholder_name}
    pattern = re.compile(r'\{(\w+)\}')

    def replacement(match):
        # Extract the placeholder name from the match object
        placeholder_name = match.group(1)

        # If the placeholder name is found in kwargs, replace it with the corresponding value
        if placeholder_name in kwargs:
            return kwargs[placeholder_name]

        # If the placeholder name is not found in kwargs, keep the original placeholder text
        return match.group(0)

    # Use the re.sub() function to replace all occurrences of the pattern in the text
    return pattern.sub(replacement, text)

def read_file(filepath):
    print(f"Reading file {filepath}.")
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "File not found."
    except PermissionError:
        return "Permission denied."
    except Exception as e:
        return f"An error occurred: {e}"

def read_file_line_by_line(filepath):
    try:
        with open(filepath, 'r') as f:
            content = []
            for line in f:
                content.append(line.strip())
            return '\n'.join(content)
    except FileNotFoundError:
        return "File not found."
    except PermissionError:
        return "Permission denied."
    except Exception as e:
        return f"An error occurred: {e}"

def get_file_size(filepath):
    try:
        return os.path.getsize(filepath)
    except FileNotFoundError:
        return 0
    except PermissionError:
        return "Permission denied."
    except Exception as e:
        return f"An error occurred: {e}"
    
def is_valid_filename(filename):
    # Normalize unicode characters
    filename = unicodedata.normalize('NFC', filename)

    # Common invalid characters across *nix and Windows
    invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
    if any(char in invalid_chars for char in filename):
        return False  # Contains invalid characters
    if len(filename.encode('utf-8')) > 255:
        return False  # Exceeds length restrictions when encoded in UTF-8
    
    # Windows-specific checks
    if platform.system() == "Windows":
        # Windows does not allow filenames to end with a dot or a space
        if filename.endswith('.') or filename.endswith(' '):
            return False
        # Check for valid drive letter
        if re.match(r'^[a-zA-Z]:\\', filename):
            return False
        # Windows reserved filenames
        reserved_names = (
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
        )
        basename, _, ext = filename.rpartition('.')
        if basename.upper() in reserved_names:
            if not ext or basename.upper() != filename.upper():
                return False

    # *nix-specific checks (optional)
    # For example, disallowing hidden files (starting with a dot)
    # if filename.startswith('.'):
    #     return False

    return True

def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                size = os.path.getsize(fp)
                total_size += size
            except OSError as e:
                print(f"Error: {e}")
                size = 0
    return total_size


def list_files_and_folders_with_sizes(start_path):
    entries = os.scandir(start_path)
    files_and_folders = []

    for entry in entries:
        # This is a check for the entry being a file or a folder at the top level only
        if entry.is_dir(follow_symlinks=False):
            entry_type = 'Folder'
            size = 0  # Do not sum up sizes within the folder
        elif entry.is_file(follow_symlinks=False):
            entry_type = 'File'
            size = get_size(entry.path)  # Get the size of the file
        else:
            entry_type = 'Other'  # For symbolic links, sockets, etc.
            size = 0  # Other types do not have a size

        files_and_folders.append({
            'name': entry.name,
            'type': entry_type,
            'size': size  # Size is in bytes
        })
    return files_and_folders

def redact_json_values(story, keys_to_redact):
    # Find all JSON objects in the string
    json_objects = re.findall(r'\{.*?\}', story, re.DOTALL)
    
    for json_obj in json_objects:
        # Load the JSON object into a Python dictionary
        try:
            data = json.loads(json_obj)
        except json.JSONDecodeError:
            continue  # Skip if it's not valid JSON
        
        # Recursive function to redact specified keys
        def redact(data):
            if isinstance(data, dict):
                for key in data:
                    if key in keys_to_redact:
                        data[key] = "..."
                    else:
                        redact(data[key])
            elif isinstance(data, list):
                for item in data:
                    redact(item)

        # Redact the necessary keys
        redact(data)
        
        # Convert the dictionary back to a JSON string
        redacted_json = json.dumps(data, indent=2)
        
        # Replace the original JSON string in the story
        story = story.replace(json_obj, redacted_json)
    
    return story

import json
import re

def redact_json_key_values_in_text(text, keys_to_redact):
    """
    Redacts the values of specified keys in JSON objects or arrays found within a text.

    Parameters:
    - text (str): The text containing potential JSON data.
    - keys_to_redact (list): A list of keys whose values should be redacted.

    Returns:
    - str: The text with redacted values in JSON objects or arrays.
    """

    def redact(data):
        """
        Recursively redacts specified keys in a JSON object or array.

        Parameters:
        - data (dict or list): The JSON object or list to redact.
        """
        if isinstance(data, dict):
            for key in data:
                if key in keys_to_redact:
                    data[key] = "[REDACTED]"
                else:
                    redact(data[key])
        elif isinstance(data, list):
            for i in range(len(data)):
                redact(data[i])

    # Regular expression pattern to match JSON objects and arrays
    json_pattern = re.compile(r'(\{.*?\}|\[.*?\])', re.DOTALL)

    # Find all JSON objects or arrays in the string
    json_strings = json_pattern.findall(text)
    
    for json_str in json_strings:
        # Load the JSON string into a Python object
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            continue  # Skip if it's not valid JSON
        
        # Redact the necessary keys
        redact(data)
        
        # Convert the object back to a JSON string
        redacted_json = json.dumps(data, indent=2)
        
        # Replace the original JSON string in the text
        text = text.replace(json_str, redacted_json)
    
    return text

def redact_json_values_in_text(text, values_to_redact):
    """
    Redacts specific values in JSON objects embedded within natural text.

    Parameters:
    - text (str): The text containing JSON objects.
    - values_to_redact (list): A list of values to be redacted.

    Returns:
    - str: The text with values redacted in JSON objects.
    """

    def redact_values(data):
        """
        Recursive function to redact values in a dictionary or list.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if value in values_to_redact:
                    data[key] = "...redacted..."
                else:
                    redact_values(value)
        elif isinstance(data, list):
            for i in range(len(data)):
                if data[i] in values_to_redact:
                    data[i] = "...redacted..."
                else:
                    redact_values(data[i])

    # Regular expression to find JSON objects in the text
    json_objects = re.findall(r'\{.*?\}', text, re.DOTALL)

    for json_obj in json_objects:
        try:
            # Convert JSON string to a Python object
            data = json.loads(json_obj)
            # Redact the specified values
            redact_values(data)
            # Convert the Python object back to a JSON string
            redacted_json = json.dumps(data, indent=2)
            # Replace the original JSON string in the text
            text = text.replace(json_obj, redacted_json)
        except json.JSONDecodeError:
            # Skip if it's not valid JSON
            continue

    return text

def get_token_count(text, token_adjust=1):
    # Define the maximum length for a text chunk
    max_length = 1000000

    # Initialize the total token count
    total_token_count = 0

    # Split the text into chunks of up to max_length characters
    for start in range(0, len(text), max_length):
        # Get a chunk of text
        chunk = text[start:start + max_length]

        # Process the chunk with the NLP tool
        doc = nlp(chunk)

        # Update the total token count
        total_token_count += int(len(doc) * token_adjust)

    # Return the total token count
    return total_token_count

token_adjust = 2.5

def trim_to_token_count(text, max_tokens):
    adjust_tokens = int(max_tokens / token_adjust)
    doc = nlp(text)
    trimmed_text = " ".join(token.text for token in doc[:adjust_tokens])
    return trimmed_text

def trim_to_right_token_count(text, max_tokens):
    adjust_tokens = int(max_tokens / token_adjust)
    doc = nlp(text)
    start = len(doc) - adjust_tokens if len(doc) > adjust_tokens else 0
    trimmed_text = " ".join(token.text for token in doc[start:])
    return trimmed_text

def trim_to_mid_token_count(text, start, max_tokens):
    adjust_tokens = int(max_tokens / token_adjust)
    doc = nlp(text)
    # Ensure start is within bounds
    start = max(0, min(len(doc) - 1, start))
    end = start + adjust_tokens
    # If max_tokens is more than the remaining tokens from start, adjust end
    end = min(len(doc), end)
    trimmed_text = " ".join(token.text for token in doc[start:end])
    return trimmed_text