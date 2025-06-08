from pydantic import BaseModel 
from tools.base_tool import BaseTool 
from typing import List, Dict, Any, Optional, Tuple 

import subprocess 
import resource 
import signal 

from prompts.tool_prompts import WRITE_FILE_TOOL_PROMPT
import tempfile
import os
import shutil



# overwrites a file if it exists, creates a new file if not 
class WriteFileToolInput(BaseModel): 
    path: str 
    content: str # full new file content  


class WriteFileToolOutput(BaseModel): 
    success: bool 

def write_file(path: str, content: str) -> bool:
    print(f'Inside write_file!')
    try:
        # Remove any leading ./ from the path
        clean_path = path.lstrip('./')
        from tools.registry import AGENT_SCRATCH_DIR
        # Build the full path within agent_scratch
        full_path = os.path.join(AGENT_SCRATCH_DIR, clean_path)
        normalized_path = os.path.normpath(full_path)

        # Prevent path traversal to protected directories and outside agent_scratch
        abs_path = os.path.abspath(normalized_path)
        agent_scratch_abs = os.path.abspath(AGENT_SCRATCH_DIR)
        if not abs_path.startswith(agent_scratch_abs):
            print(f"Error: Attempted path traversal or writing outside agent_scratch: {abs_path}")
            return False

        # Limit file size (1MB like in run_code_tool)
        if len(content) > 1024 * 1024:
            print(f"Error: File content too large. Max size: 1MB")
            return False
        
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(abs_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        # Write the file
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Successfully wrote file: {abs_path}")
        return True
        
    except PermissionError:
        print(f"Error: Permission denied writing to {path}")
        return False
    except OSError as e:
        print(f"Error: OS error writing to {path}: {e}")
        return False
    except Exception as e:
        print(f"Error: Unexpected error writing to {path}: {e}")
        return False


class WriteFileTool(BaseTool): 
    def __init__(
        self, 
        name: str = "write_file", 
        description: str = "Write to or edit an existing file", 
        prompt: str = WRITE_FILE_TOOL_PROMPT, 
        input_type: type = WriteFileToolInput, 
    ):
        super().__init__(name, description, prompt, input_type)

    def _execute(self, tool_input: WriteFileToolInput) -> WriteFileToolOutput: 
        # Check if file extension is allowed
        allowed_extensions = [".txt", ".md", ".py", ".sh", ".json", ".yaml", ".yml", ".html", ".css", ".js"]

        if not any(tool_input.path.endswith(ext) for ext in allowed_extensions): 
            print(f"Error: File extension not allowed for path: {tool_input.path}")
            success = False 
        else: 
            success = write_file(tool_input.path, tool_input.content)
        return WriteFileToolOutput(
            success=success, 
        )