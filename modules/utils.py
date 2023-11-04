import json
import os
import platform
from termcolor import colored
import spacy
import re
from rich.console import Console

console = Console()

# Load English tokenizer, POS tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

def get_token_count(text):
    doc = nlp(text)
    return len(doc) * 2

# Example usage:
# text = "Hello, how are you?"
# token_count = get_token_count(text)
# print(token_count)  # Output: 5

def trim_to_token_count(text, max_tokens):
    adjust_tokens = int(max_tokens / 2)
    doc = nlp(text)
    trimmed_text = " ".join(token.text for token in doc[:adjust_tokens])
    return trimmed_text

def trim_to_right_token_count(text, max_tokens):
    adjust_tokens = int(max_tokens / 2)
    doc = nlp(text)
    start = len(doc) - adjust_tokens if len(doc) > adjust_tokens else 0
    trimmed_text = " ".join(token.text for token in doc[start:])
    return trimmed_text

def trim_to_mid_token_count(text, start, max_tokens):
    adjust_tokens = int(max_tokens / 2)
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
    windows_wanted_extensions = ['.exe', '.bat']
    
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

def print_colored_text_old(text, end_newline=True):
    """
    Utility function to print colored and styled text to the console.
    The text can contain special formatting tags like [bold], [italic], and [c:<color>].
    Example: "[bold][c:green]Hello\n[c:magenta][italic]World!"
    """
    # Define color and style mappings
    color_map = {
        'red': 'red',
        'green': 'green',
        'yellow': 'yellow',
        'blue': 'blue',
        'magenta': 'magenta',
        'cyan': 'cyan',
        'white': 'white'
    }
    style_map = {
        'bold': 'bold',
        'underline': 'underline'
    }
    
    # Initialize variables to hold the current color and styles
    current_color = None
    current_styles = []
    
    # Split the text by the [ tag to identify formatting commands
    parts = text.split("[")
    
    # Initialize an empty string to hold the final styled text
    styled_text = ""
    
    for part in parts:
        if part.startswith("c:"):
            # Set color
            color_code = part[2:].split("]")[0]
            current_color = color_map.get(color_code, None)
        elif part.split("]")[0] in style_map:
            # Add style
            style_cmd = part.split("]")[0]
            current_styles.append(style_map[style_cmd])
        elif "]" in part:
            # Reset styles or color
            reset_cmds = part.split("]")
            for cmd in reset_cmds:
                if cmd == "reset":
                    current_styles = []
                    current_color = None
                elif cmd == "nobold":
                    current_styles.remove('bold')
                elif cmd == "nounderline":
                    current_styles.remove('underline')
                elif cmd.startswith("c:"):
                    # Reset color only
                    current_color = None
        else:
            # This part is a text that needs to be styled
            styled_text = colored(part, current_color, attrs=current_styles)
            print(styled_text, end="", flush=True)  # flush stdout here
    
    # Print a newline character to end the line
    if end_newline:
        print()


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

# Example usage:
# text = 'Return only commands listed in the {run_command_list} section for {get_os_name}\'s consoles.'
# kwargs = {
#     'run_command_list': '...',
#     'get_os_name': 'Windows'
# }
# replaced_text = replace_placeholders(text, **kwargs)
# print(replaced_text)

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