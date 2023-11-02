# ShellSpeak
![ShellSpeak Logo](assets/logo.png)
ShellSpeak is an interactive command-line interface that enhances the terminal experience by integrating AI-driven command translation and execution. The core functionality of ShellSpeak revolves around capturing user input, translating it to actionable shell commands through an AI model, and executing these commands while displaying the output in a styled and user-friendly manner.

## Notice
- This can and will delete files if you are not careful, I suggest you use this on a system you do not care about, or a emulator.
- I only have developed and tested on windows, "should" work with other consoles, and on windows, OpenAI loves using *nix comments.
- Feel free to look at what I have done, and please check back in a day or two.
- I plan on Expanding on this README this is just a quick draft.

## Highlights
- Works just like the console. (With pretty colors.)
- Embed files into you commands by using "file: /filepath/" (where "/filepath/" is the path to your file, allows for multiple files.)
- Uses past conversation history.
- Checks your PATH for all available commands, and implies default OS commands.
- Use plain text for commands i.e. "Go to folder Test"
- The modules/llm.py file has our LLM class, that allows for offline (huggingface.co transformers) and OpenAI (gpt3/gpt4), and can be easily expanded on.

(Files/History/PATH commands, are set to set token limits that are can be adjusted, check modules/shellSpeak.py)

## Information

![ShellSpeak Command Processing Flow](assets/flow.png)

The above diagram illustrates the architecture of ShellSpeak. It provides an in-depth look into how the program captures and processes user input to execute commands. We have recently added the capability to handle file inputs and to manage command history, enhancing the overall user experience.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/TheCompAce/ShellSpeak
    cd ShellSpeak
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Configure your settings in `settings.json` to match your preferences and system setup.

## Usage

Run the `ShellSpeak` script from the command line:
```bash
python main.py
```

## Contribute
- Check our Disscussion on our [GitHub Discussion](https://github.com/TheCompAce/ShellSpeak/discussions)
- You can join to help build the code base, or to mod the GitHub. (I do have another job, this is mainly for learning.)
- You can also donate (@BradfordBrooks79 on Venmo) if you want (I would love to do stuff like this full time.)

## License

ShellSpeak is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

