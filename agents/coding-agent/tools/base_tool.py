from pydantic import BaseModel # for data and types
from abc import ABC, abstractmethod # for behavior/methods
from typing import List, Union, Optional, Dict, Any, Tuple
import re 
import json  


class BaseTool(ABC): 
    def __init__(self, name: str, description: str, prompt: str, input_type: type): 
        self.name = name 
        self.description = description 
        self.prompt = prompt 
        self.input_type = input_type 

    @abstractmethod
    def _execute(self, input: BaseModel) -> BaseModel: 
        pass 

    # same for all 
    def _parse(self, s: str) -> List[Tuple[BaseModel, int, int]]: 
        # pattern = rf'<tool_request>\s*\{{[^{{}}]*"tool_name"\s*:\s*"{self.name}"[^{{}}]*\}}\s*</tool_request>' 
        pattern = rf'<tool_request>(.*?)</tool_request>'
        matches = list(re.finditer(pattern, s, re.DOTALL))
        
        out = []

        if not matches: 
            return None 

        for match in reversed(matches): 
            try: 
                # extract just the JSON from the match (between the XML tags)
                # think something weird happening here -- all models bug out with my prompt 
                inner = match.group(1).strip()
                # first_line = next((ln for ln in inner.splitlines() if ln.strip()), "")
                # json_data = json.loads(first_line)
                json_data = json.loads(inner)

                if json_data.get("tool_name") != self.name: continue # meant for another tool 

                # note the string indices of start/finish for the entire XML block
                start_pos = match.start()
                end_pos = match.end()
                # add to list 
                input_obj = self.input_type(**json_data)
                out.append((input_obj, start_pos, end_pos))
            except: 
                print(f'FAILED PARSING JSON INTO OBJECT FOR {self.name}, json={match.group()}')
                continue 
        return out 

    def call(self, s: str) -> str: 
        while True:
            res = self._parse(s)
            if not res: # no more tool calls 
                break

            for parsed in res: 
                if parsed is None: 
                    print(f'WARNING: Failed parsing tool call query from LLM in SearchTool.call()...')
                    continue 
                else: 
                    tool_input, start, end = parsed
                    print(f'[Calling tool: {self.name}...]')
                    tool_output_json = self._execute(tool_input).model_dump()
                    output_s = json.dumps(tool_output_json)
                    wrapped_output_s = f"<tool_output name='{self.name}' request='{json.dumps(tool_input.model_dump())}'>" + output_s + f"</tool_output>"
                    s = s[:start] + wrapped_output_s + s[end:]

        return s