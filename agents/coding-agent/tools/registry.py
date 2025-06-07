# some tools we aim to support: [search (shallow and deep), readFile, writeFile, runCode, thinkHard, askBigBro]

# to make a new tool: 
    # create a class new_tool.py defining the functionality (input/output types, execution logic)
    # add a corresponding prompt in tool_prompts.py 
    # list the new class and prompt below in this file 

from tools.search_tool import SearchTool
from tools.run_code_tool import RunCodeTool
from tools.write_file_tool import WriteFileTool
from tools.read_file_tool import ReadFileTool
from prompts.tool_prompts import SEARCH_TOOL_PROMPT, RUN_CODE_TOOL_PROMPT, WRITE_FILE_TOOL_PROMPT, READ_FILE_TOOL_PROMPT

AGENT_SCRATCH_DIR = "./agent_scratch/"

ALL_TOOLS = [
    SearchTool(), # instantiate each so we can use tool in all_tools directly out of the box 
    RunCodeTool(), 
    WriteFileTool(), 
    ReadFileTool(), 
]

ALL_TOOL_PROMPTS = [
    SEARCH_TOOL_PROMPT,
    RUN_CODE_TOOL_PROMPT, 
    WRITE_FILE_TOOL_PROMPT, 
    READ_FILE_TOOL_PROMPT, 
]