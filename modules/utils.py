import json
import os
import platform
import unicodedata
import spacy
import re
from rich.console import Console

from functools import partial
from multiprocessing import Pool, TimeoutError

console = Console()

token_adjust = 2.5

# Load English tokenizer, POS tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

def get_token_count(text):
    doc = nlp(text)
    return int(len(doc) * token_adjust)

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
    end = "\n" if end_newline else ""
    console.print(text, end=end)

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