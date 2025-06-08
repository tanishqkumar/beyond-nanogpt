from pydantic import BaseModel 
from tools.base_tool import BaseTool 
from typing import List, Dict, Any, Optional, Tuple 

import subprocess 
import resource 
import signal 

from prompts.tool_prompts import READ_FILE_TOOL_PROMPT
import tempfile
import os
import shutil

# Reads files from the local filesystem (no agent_scratch restriction)
class ReadFileToolInput(BaseModel): 
    path: str 

class ReadFileToolOutput(BaseModel): 
    success: bool 
    content: str = ""

def read_file(path: str) -> Tuple[bool, str]:
    try:
        # Normalize the path and ensure it's relative or absolute as given
        # normalized_path = os.path.normpath(path)
        clean_path = path.lstrip('./')
        from tools.registry import AGENT_SCRATCH_DIR
        # Build the full path within agent_scratch
        full_path = os.path.join(AGENT_SCRATCH_DIR, clean_path)
        normalized_path = os.path.normpath(full_path)
        
        # Prevent path traversal to protected directories
        protected_dirs = ['/etc', '/usr', '/bin', '/sbin', '/sys', '/proc', '/dev']
        abs_path = os.path.abspath(normalized_path)
        for protected in protected_dirs:
            if abs_path.startswith(protected):
                print(f"Error: Cannot read from protected directory: {abs_path}")
                return False, ""

        # read the file
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Successfully read file: {abs_path}")
        return True, content

    except FileNotFoundError:
        print(f"Error: File not found: {path}")
        return False, ""
    except PermissionError:
        print(f"Error: Permission denied reading from {path}")
        return False, ""
    except OSError as e:
        print(f"Error: OS error reading from {path}: {e}")
        return False, ""
    except Exception as e:
        print(f"Error: Unexpected error reading from {path}: {e}")
        return False, ""


class ReadFileTool(BaseTool): 
    def __init__(
        self, 
        name: str = "read_file", 
        description: str = "Read the contents of an existing file", 
        prompt: str = READ_FILE_TOOL_PROMPT, 
        input_type: type = ReadFileToolInput, 
    ):
        super().__init__(name, description, prompt, input_type)

    def _execute(self, tool_input: ReadFileToolInput) -> ReadFileToolOutput: 
        success, content = read_file(tool_input.path)
        return ReadFileToolOutput(
            success=success, 
            content=content, 
        )