# Replaces the built-in function for Input, so ShellSpeak can handle this.
import builtins

original_input = builtins.input

def custom_input(prompt=None):
    # Process the prompt argument here
    print(f'{prompt}')    
    print(f':WAIT_FOR_INPUT:')

    return_input = original_input(prompt)
    return return_input

# Replace the built-in input function
builtins.input = custom_input

# Now when you call input, it will be routed through custom_input
# user_input = input('Enter something: ')


COMMAND: cd ..
