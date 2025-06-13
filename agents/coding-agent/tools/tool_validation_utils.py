import json  
import re
from typing import List, Tuple

def is_valid_tool_call(call_json: str) -> bool:
    """Check if a tool call JSON string is valid."""
    # whitelist of supported tools and their parameter schemas
    valid_tools = {
        "search": {"required": ["query"], "optional": ["top_k", "deep", "max_len"]},
        "run_code": {"required": ["code"], "optional": ["language"]},
        "write_file": {"required": ["path", "content"], "optional": []},
        "read_file": {"required": ["path"], "optional": []}
    }
    
    try:
        call = json.loads(call_json)
        
        # basic structure validation
        if "tool_name" not in call:
            return False
        
        tool_name = call["tool_name"]
        
        # empty tool name is invalid
        if tool_name == "":
            return False
        
        # unknown tool
        if tool_name not in valid_tools:
            return False
        
        # check all required params are present
        required_params = valid_tools[tool_name]["required"]
        for param in required_params:
            if param not in call:
                return False
        
        # check no unexpected params are present
        valid_params = set(required_params + valid_tools[tool_name]["optional"] + ["tool_name"])
        for param in call.keys():
            if param not in valid_params:
                return False
        
        return True
        
    except json.JSONDecodeError:
        # malformed json
        return False


def get_invalid_positions(s: str) -> List[Tuple[int, int]]: 
    # find all tool_request blocks and check if they're valid
    pattern = r'<tool_request>\s*(.*?)\s*</tool_request>'
    matches = re.finditer(pattern, s, re.DOTALL)
    
    invalids = []
    
    for match in matches:
        if not is_valid_tool_call(match.group(1)):
            invalids.append((match.start(), match.end()))
    
    return invalids

def check_and_fix_tool_calls(s: str) -> str:
    # models often generate malformed json with unescaped newlines/quotes
    def replace_special_chars_in_braces(text: str) -> str:
        result = []
        in_braces = 0
        in_string = False
        prev_char = ''
        
        for c in text:
            if c == '{' and not in_string:
                in_braces += 1
                result.append(c)
            elif c == '}' and not in_string:
                if in_braces > 0:
                    in_braces -= 1
                result.append(c)
            elif c == '"' and prev_char != '\\':
                in_string = not in_string
                result.append(c)
            elif in_braces > 0 and in_string:
                # escape special chars inside json string values
                if c == '\n':
                    result.append('\\n')
                elif c == '\r':
                    result.append('\\r')
                elif c == '\t':
                    result.append('\\t')
                elif c == '"' and prev_char != '\\':
                    result.append('\\"')
                else:
                    result.append(c)
            else:
                result.append(c)
            
            prev_char = c
            
        return ''.join(result)

    return replace_special_chars_in_braces(s)

def contains_tool_calls(s: str) -> bool: 
    # simple check for any tool request tags
    return bool('<tool_request>' in s)