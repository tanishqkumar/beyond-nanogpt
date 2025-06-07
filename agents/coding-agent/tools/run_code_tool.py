from pydantic import BaseModel 
from tools.base_tool import BaseTool 
from typing import List, Dict, Any, Optional, Tuple 

import subprocess 
import resource 
import signal 

from prompts.tool_prompts import RUN_CODE_TOOL_PROMPT
from prompts.sandbox_prompts import PYTHON_SANDBOX_PROMPT, SHELL_SANDBOX_PROMPT
import tempfile
import os
import shutil

class RunCodeToolInput(BaseModel): 
    code: str 
    language: str = "python"  # python or shell only

class RunCodeToolOutput(BaseModel): 
    stdout: str = ""
    stderr: str = ""


def run_code(code: str, language: str) -> Tuple[str, str]:
    # inherit environment but run in subprocess for isolation
    env = os.environ.copy()
    try: 
        if language == "python": 
            # wrap user code in sandbox template that restricts imports
            code = PYTHON_SANDBOX_PROMPT.format(
                code=code,
            )

            res = subprocess.run(
                ["python3", "-c", code], 
                capture_output=True, 
                text=True, 
                timeout=10,  # prevent infinite loops
                env=env,
            )

        elif language == "shell": 
            # wrap shell commands in sandbox template
            code = SHELL_SANDBOX_PROMPT.format(
                code=code,
            )

            res = subprocess.run(
                code, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=10,  # prevent infinite loops
                env=env,
            )
        else:
            return "", f"Unsupported language: {language}"
        return res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return "", "Error: Agent code timed out..."
    except Exception as e:
        return "", f"Error: {str(e)}"
    
    


class RunCodeTool(BaseTool):
    def __init__(
        self, 
        name: str = "run_code", 
        description: str = "Execute code on small cpu and return (stdout, stderr)", 
        prompt: str = RUN_CODE_TOOL_PROMPT, 
        input_type: type = RunCodeToolInput, 
    ):
        super().__init__(name, description, prompt, input_type)

    def _execute(self, tool_input: RunCodeToolInput) -> RunCodeToolOutput: 
        stdout, stderr = run_code(tool_input.code, tool_input.language)
        return RunCodeToolOutput(
            stdout=stdout, 
            stderr=stderr, 
        )
