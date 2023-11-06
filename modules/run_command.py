import asyncio
from asyncore import loop
import os

class CommandRunner:
    def __init__(self, shell_speak):
        self.shell_speak = shell_speak
        self.collected_output = ""
        self.pause_time = 1.0
        # self.output_event = asyncio.Event()
        self.use_input = False

    async def run(self, command):
        self.collected_output = ""

        my_error = {
            "err": False,
            "desc": ""
        }
        
        process = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        async def read_stream(stream, callback):
            while True:
                await asyncio.sleep(self.pause_time) 
                line = await stream.readline()
                self.use_input = False
                if line:
                    if line != b'':
                        decode_line = line.decode('utf-8').strip()
                        if decode_line != ":WAIT_FOR_INPUT:":
                            # self.collected_output += decode_line
                            callback(decode_line)
                        else:
                            self.use_input = True

                # Check if the process is still running
                return_code = process.returncode  # None if the process is still running
                if return_code is not None:
                    # The process has finished, so just return the collected output
                    break

        async def write_stream():
            # Allow some time for the process to complete
            await asyncio.sleep(self.pause_time) 
            
            while True:
                try:
                    # Wait for a short period to see if new output arrives
                    await asyncio.sleep(self.pause_time) 

                    # Check if the process is still running
                    return_code = process.returncode  # None if the process is still running
                    if return_code is not None:
                        # The process has finished, so just return the collected output
                        break

                    # await self.output_event.wait()
                    # self.output_event.clear()

                    # Check for new output again
                    if self.collected_output:
                        translated_output = self.shell_speak.translate_output(self.collected_output, True)
                        self.shell_speak.display_output(translated_output)
                        self.collected_output = ""
                        
                    # No new output, so prompt for user input
                    user_input = None
                    if self.use_input:
                        user_input = await asyncio.to_thread(input, self.collected_output)
                        self.use_input = False

                    self.collected_output = ""
                    
                    if user_input:
                        process.stdin.write(user_input.encode() + b'\n')
                    else:  # Assume EOF if input is empty
                        break
                except EOFError:
                    # Handle Ctrl-Z (EOF) to cancel if needed
                    my_error["err"] = True
                    my_error["desc"] = "Ctrl-Z"
                    print("Ctrl-Z detected, exiting...")
                    break
                except Exception as e:
                    # Log or handle other exceptions
                    my_error["err"] = True
                    my_error["desc"] = e
                    break  # Optionally break out of the loop on any exception

                # Optionally add a delay to avoid busy-waiting
                # await asyncio.sleep(0.1)


        await asyncio.gather(
            read_stream(process.stdout, self.handle_stdout),
            read_stream(process.stderr, self.handle_stderr),
            write_stream()
        )

        stdout, stderr = await process.communicate()

        if my_error["err"]:
            stderr = my_error["desc"]

        return self.collected_output, stderr.decode('utf-8') if not my_error["err"] else stderr


    def handle_stdout(self, line):
        if line.strip() != "" and line != ":WAIT_FOR_INPUT:":
            self.collected_output += line + "\n"

    def handle_stderr(self, line):
        formatted_error = self.shell_speak.translate_output(line, True)
        self.shell_speak.display_output(formatted_error)
