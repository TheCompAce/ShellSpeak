import os
import platform
from termcolor import colored
import spacy

# Load English tokenizer, POS tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

def get_token_count(text):
    doc = nlp(text)
    return len(doc)

# Example usage:
# text = "Hello, how are you?"
# token_count = get_token_count(text)
# print(token_count)  # Output: 5

def trim_to_token_count(text, max_tokens):
    doc = nlp(text)
    trimmed_text = " ".join(token.text for token in doc[:max_tokens])
    return trimmed_text

def get_os_name():
    return platform.system()

def map_possible_commands():
    # Get the PATH environment variable
    path_variable = os.environ.get('PATH', '')
    
    # Split it into individual directories
    directories = path_variable.split(os.pathsep)
    
    # Initialize a set to store unique command names
    unique_commands = set()
    
    for directory in directories:
        try:
            # List all files in the directory
            files = os.listdir(directory)
            
            # Filter out executable files and add them to the set
            for file in files:
                file_path = os.path.join(directory, file)
                if os.access(file_path, os.X_OK):
                    unique_commands.add(file)
        except FileNotFoundError:
            # Directory in PATH does not exist, skip
            continue
        except PermissionError:
            # Don't have permission to access directory, skip
            continue
    commands_str = ' '.join(unique_commands)
    return commands_str

def print_colored_text(text, end_newline=True):
    """
    Utility function to print colored and styled text to the console.
    The text can contain special formatting tags like /*bold/*, /*italic/*, and /*c:<color>/*.
    Example: "/*bold/*/*c:green/*Hello\n/*c:magenta/*/*italic/*World!"
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
    
    # Split the text by the /* tag to identify formatting commands
    parts = text.split("/*")
    
    # Initialize an empty string to hold the final styled text
    styled_text = ""
    
    for part in parts:
        if part.startswith("c:"):
            # Set color
            color_code = part[2:].split("/*")[0]
            current_color = color_map.get(color_code, None)
        elif part in style_map:
            # Add style
            current_styles.append(style_map[part])
        elif "/*" in part:
            # Reset styles or color
            reset_cmds = part.split("/*")
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
            print(styled_text, end="")
    
    # Print a newline character to end the line
    if end_newline:
        print()

def capture_styled_input(prompt):
    # Print the prompt without a newline at the end
    print_colored_text(prompt, end_newline=False)
    
    # Capture and return user input
    return input()

def replace_placeholders(text, **kwargs):
    """
    Replaces placeholders in the given text with the values provided.

    Parameters:
    - text (str): The text containing placeholders.
    - **kwargs: The values to replace the placeholders with.

    Returns:
    - str: The text with placeholders replaced.
    """
    try:
        return text.format(**kwargs)
    except KeyError as e:
        return f"Placeholder {e} not found in keyword arguments"

# Usage:
# text = "Hello, {name}! Welcome to {place}."
# replaced_text = replace_placeholders(text, name="Brad", place="North Carolina")
# print(replaced_text)  # Output: Hello, Brad! Welcome to North
