import asyncio
from asyncore import loop
import os

class CommandRunner:
    def __init__(self, shell_speak):
        self.shell_speak = shell_speak
        self.collected_output = ""
        self.collected_history = ""
        self.pause_time = 0.5
        # self.output_event = asyncio.Event()
        self.use_input = False

    async def run(self, command):
        self.collected_output = ""
        self.collected_history = ""

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

        async def read_lines(stream):
            lines = []
            while True:
                line = None
                # print("OK T1")
                line = await stream.readline()
                if not line:
                    # print("OK T2")
                    break

                # print(f"line = {line}")
                lines.append(line)

                # await asyncio.sleep(self.pause_time * 2)
            return lines

        async def read_stream(stream, callback):
            while True:
                await asyncio.sleep(self.pause_time) 
                lines = await read_lines(stream)
                for line in lines:
                    self.use_input = False
                    # print("OK9")
                    if line:
                        if line != b'':
                            decode_line = line.decode('utf-8').strip()
                            if decode_line != ":WAIT_FOR_INPUT:":
                                self.collected_output += "\n" + decode_line
                                # callback(decode_line)
                                self.collected_history += "\n" + decode_line
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
                        print(f"self.collected_output = {self.collected_output}")
                        translated_output = self.shell_speak.translate_output(self.collected_output, True)
                        print(f"translated_output = {translated_output}")
                        self.shell_speak.display_output(translated_output)
                        print("OK2")
                        self.collected_history += "\n" + self.collected_output
                        self.collected_output = ""
                        
                    # No new output, so prompt for user input
                    user_input = None
                    if self.use_input:
                        user_input = await asyncio.to_thread(input, self.collected_output)
                        self.use_input = False
                        self.collected_output = ""
                    
                        if user_input:
                            process.stdin.write(user_input.encode() + b'\n')
                        else:
                            process.stdin.close()  # Signal EOF to the subprocess
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

        # await asyncio.sleep(self.pause_time) 
        # stdout, stderr = await process.communicate()

        stderr = ""

        if my_error["err"]:
            stderr = my_error["desc"]

        # print(f"self.collected_history = {self.collected_history}")
        return self.collected_history, stderr if not my_error["err"] else stderr


    def handle_stdout(self, line):
        print(f"line = {line}")
        if line.strip() != "" and line != ":WAIT_FOR_INPUT:":
            print("OK 1")
            self.collected_history += line + "\n"
            self.collected_output += line + "\n"

    def handle_stderr(self, line):
        print("OK E")
        formatted_error = self.shell_speak.translate_output(line, True)
        self.shell_speak.display_output(formatted_error)
