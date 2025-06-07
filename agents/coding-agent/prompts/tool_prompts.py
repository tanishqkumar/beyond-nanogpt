import os

SEARCH_TOOL_PROMPT = """
    ----- 
    SEARCH TOOL - Use for real-time web information and factual data.
    
    FORMAT: 
    <tool_request>
    {"tool_name": "search", "query": "search_query", "top_k": 3, "deep": false, "max_len": 20000}
    </tool_request>
    
    PARAMETERS:
    - query (required): Search query string
    - top_k (optional, default 3): Number of results (1-10)
    - deep (optional, default false): false = shallow (snippets), true = deep (full page contents)
    - max_len (optional, default 20000): Max character limit for results
    
    EXAMPLES:
    
    <tool_request>
    {"tool_name": "search", "query": "Bitcoin price current USD", "top_k": 2}
    </tool_request>
    <tool_request>
    {"tool_name": "search", "query": "weather forecast New York today"}
    </tool_request>
    <tool_request>
    {"tool_name": "search", "query": "Python tutorial", "deep": true}
    </tool_request>
    <tool_request>
    {"tool_name": "search", "query": "quantum computing applications 2024", "top_k": 5, "deep": true}
    </tool_request>
    
    COMMON MISTAKES TO AVOID (NON-EXAMPLES):
    
    ❌ WRONG - Invalid tool_name name:
    <tool_request>{"tool_name": "web_search", "query": "Bitcoin price"}</tool_request>
    
    ❌ WRONG - Using "text" instead of "query":
    <tool_request>{"tool_name": "search", "text": "Bitcoin price current USD"}</tool_request>
    
    ❌ WRONG - Invalid wrapping tags:
    <tool_name>{"tool_name": "search", "query": "NFL playoff results 2024"}</tool_name>
    
    ❌ WRONG - Invalid JSON syntax (single quotes):
    <tool_request>{'tool_name': 'search', 'query': 'Python tutorial'}</tool_request>
    
    ❌ WRONG - Missing query parameter:
    <tool_request>{"tool_name": "search", "top_k": 5}</tool_request>
    
    ❌ WRONG - Unescaped double quotes in value:
    <tool_request>{"tool_name": "search", "query": "What is a "quantum computer"?"}</tool_request>
    
    ❌ WRONG - Trailing comma in JSON:
    <tool_request>{"tool_name": "search", "query": "AI news",}</tool_request>
    
    ❌ WRONG - Newlines in value not properly escaped:
    <tool_request>{"tool_name": "search", "query": "First line
    Second line"}</tool_request>
    
    ❌ WRONG - Backslash not properly escaped:
    <tool_request>{"tool_name": "search", "query": "C:\new_folder\file.txt"}</tool_request>
    
    TOOL INTERACTIONS & WORKFLOWS:
    - Use search results to inform other tool usage (e.g., search for documentation, then write_file to save code)
    - Multiple searches are encouraged for multi-hop information gathering
    - Search can complement run_code for research-driven development
    
    IMPORTANT: Never claim you "searched" or "used the search tool" unless you actually made a <tool_request> with the search tool.
    
    GUIDELINES:
    - Use shallow (deep: false) for quick facts, current events
    - Use deep (deep: true) for tutorials, detailed explanations, involved references
    - Use specific, precise search terms
    - JSON must be valid - use double quotes
    - Always wrap in <tool_request></tool_request> tags
    - Tool name MUST be "search" and the search query MUST go in "query" field
    - Multiple searches are perfectly fine for comprehensive research
    - For any files you read or write (including code files in coding projects), always use the ./agent_scratch/ directory.
    ----- 
"""



# you'll need to set these in the env
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "")
GITHUB_PAT = os.getenv("GITHUB_PAT", "")
GITHUB_EMAIL = os.getenv("GITHUB_EMAIL", "")
GITHUB_NOREPLY_EMAIL = os.getenv("GITHUB_NOREPLY_EMAIL", "")

RUN_CODE_TOOL_PROMPT = """
    ----- 
    RUN CODE TOOL - Execute Python or shell code safely in a sandboxed environment.
    
    FORMAT: 
    <tool_request>
    {"tool_name": "run_code", "code": "code_goes_here", "language": "python"}
    </tool_request>
    
    PARAMETERS:
    - code (required): Code string to execute (max 100 lines)
    - language (optional, default "python"): "python" or "shell" -- NO other languages supported
    
    EXECUTION ENVIRONMENT:
    - Runs in isolated sandbox/container 
    - Fresh container for each execution - no state persistence
    - Limited execution time: 10 seconds timeout
    - Resource limits: 100MB memory, 10 second CPU time, 1MB file size
    - Working directory: /tmp/user_code_exec/ (isolated from host system)
    - No internet access or external network connections
    - You have access to a GitHub account/credentials to manage version control 
    
    AVAILABLE PYTHON IMPORTS:
    ✅ Standard Library (safe subset): math, random, datetime, json, csv, re, itertools, collections, statistics, decimal, fractions, typing, dataclasses, enum
    ❌ BLOCKED: os, sys, subprocess, socket, urllib.request, requests, threading, multiprocessing, any third-party packages
    
    EXAMPLES:
    
    Python Examples:
    <tool_request>
    {"tool_name": "run_code", "code": "print('Hello World!')"}
    </tool_request>
    <tool_request>
    {"tool_name": "run_code", "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\nprint(factorial(5))", "language": "python"}
    </tool_request>
    <tool_request>
    {"tool_name": "run_code", "code": "import json  \ndata = {'name': 'AI', 'tasks': ['search', 'code', 'analyze']}\nprint(json.dumps(data, indent=2))"}
    </tool_request>
    
    Shell Examples:
    <tool_request>
    {"tool_name": "run_code", "code": "echo 'Hello from shell'", "language": "shell"}
    </tool_request>
    <tool_request>
    {"tool_name": "run_code", "code": "for i in {1..5}; do echo \"Count: $i\"; done", "language": "shell"}
    </tool_request>
    <tool_request>
    {"tool_name": "run_code", "code": "ls -la", "language": "shell"}
    </tool_request>
    <tool_request>
    {"tool_name": "run_code", "code": "pwd && find . -type f -name '*.txt' | head -10", "language": "shell"}
    </tool_request>
    
    COMMON MISTAKES TO AVOID (NON-EXAMPLES):
    
    ❌ WRONG - Invalid tool_name name:
    <tool_request>{"tool_name": "execute", "code": "print('test')"}</tool_request>
    
    ❌ WRONG - Using "script" instead of "code":
    <tool_request>{"tool_name": "run_code", "script": "print('test')"}</tool_request>
    
    ❌ WRONG - Invalid JSON syntax (single quotes):
    <tool_request>{'tool_name': 'run_code', 'code': 'print("test")'}</tool_request>
    
    ❌ WRONG - Using unsupported language:
    <tool_request>{"tool_name": "run_code", "code": "console.log('hello')", "language": "javascript"}</tool_request>
    
    ❌ WRONG - Expecting third-party libraries:
    <tool_request>{"tool_name": "run_code", "code": "import numpy as np\nprint(np.array([1,2,3]))"}</tool_request>
    
    ❌ WRONG - Unescaped newlines in code:
    <tool_request>{"tool_name": "run_code", "code": "print('Hello'\n'World!')"}</tool_request>
    
    ❌ WRONG - Unescaped double quotes in code:
    <tool_request>{"tool_name": "run_code", "code": "print("Hello")"}</tool_request>
    
    ❌ WRONG - Trailing comma in JSON:
    <tool_request>{"tool_name": "run_code", "code": "print('test')",}</tool_request>
    
    ❌ WRONG - Backslash not properly escaped:
    <tool_request>{"tool_name": "run_code", "code": "with open('C:\temp\file.txt') as f: print(f.read())"}</tool_request>

    
    SECURITY RESTRICTIONS:
    - Code executes in sandboxed container with 10-second timeout
    - NO file I/O outside isolated directory, NO network access (except GitHub via git/curl), NO system commands
    - NO dangerous imports (os, sys, subprocess), NO third-party packages
    - Resource limits: 100MB memory, 10 second CPU time, 1MB file size
    - Fresh container for each execution - no persistence between runs
    
    DEBUGGING & ITERATION:
    - Tool returns both stdout and stderr for error analysis
    - Feel free to iteratively debug and improve code until it runs correctly
    - Each execution is independent - no variables persist between runs

    GITHUB INTEGRATION:
    You have access to GitHub under username "{GITHUB_USERNAME}" with this personal access token:
    {GITHUB_PAT}
    In case you need it, the email is {GITHUB_EMAIL} 
    
    CRITICAL: When working with Git/GitHub, ALWAYS actually execute the commands using run_code with shell language.
    Do NOT just print or simulate git commands - ACTUALLY RUN THEM.
    
    GitHub Setup Commands (run these first when working with GitHub):
    <tool_request>
    {{"tool_name": "run_code", "code": "git config --global user.name '{GITHUB_USERNAME}'", "language": "shell"}}
    </tool_request>
    <tool_request>
    {{"tool_name": "run_code", "code": "git config --global user.email '{GITHUB_NOREPLY_EMAIL}'", "language": "shell"}}
    </tool_request>
    
    GitHub Authentication (use PAT for HTTPS operations):
    - Clone: git clone https://{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/repo-name.git
    - Push: git push https://{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/repo-name.git
    
    GitHub Workflow Examples:
    
    Create New Repository:
    <tool_request>
    {{"tool_name": "run_code", "code": "curl -X POST -H 'Authorization: token {GITHUB_PAT}' -H 'Content-Type: application/json' -d '{{\"name\":\"my-new-repo\",\"description\":\"My project\",\"private\":false}}' https://api.github.com/user/repos", "language": "shell"}}
    </tool_request>
    
    Initialize and Push to GitHub:
    <tool_request>
    {"tool_name": "run_code", "code": "cd ./agent_scratch && git init", "language": "shell"}
    </tool_request>
    <tool_request>
    {"tool_name": "run_code", "code": "cd ./agent_scratch && git add .", "language": "shell"}
    </tool_request>
    <tool_request>
    {"tool_name": "run_code", "code": "cd ./agent_scratch && git commit -m 'Initial commit'", "language": "shell"}
    </tool_request>
    <tool_request>
    {"tool_name": "run_code", "code": "cd ./agent_scratch && git branch -M main", "language": "shell"}
    </tool_request>
    <tool_request>
    {{"tool_name": "run_code", "code": "cd ./agent_scratch && git remote add origin https://{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/my-new-repo.git", "language": "shell"}}
    </tool_request>
    <tool_request>
    {"tool_name": "run_code", "code": "cd ./agent_scratch && git push -u origin main", "language": "shell"}
    </tool_request>
    
    Clone Existing Repository:
    <tool_request>
    {{"tool_name": "run_code", "code": "cd ./agent_scratch && git clone https://{GITHUB_PAT}@github.com/{GITHUB_USERNAME}/existing-repo.git", "language": "shell"}}
    </tool_request>
    
    Common Git Operations:
    <tool_request>
    {"tool_name": "run_code", "code": "cd ./agent_scratch && git status", "language": "shell"}
    </tool_request>
    <tool_request>
    {"tool_name": "run_code", "code": "cd ./agent_scratch && git add . && git commit -m 'Update files'", "language": "shell"}
    </tool_request>
    <tool_request>
    {"tool_name": "run_code", "code": "cd ./agent_scratch && git push", "language": "shell"}
    </tool_request>
    <tool_request>
    {"tool_name": "run_code", "code": "cd ./agent_scratch && git pull", "language": "shell"}
    </tool_request>
    
    
    ❌ WRONG - Just printing git commands instead of running them:
    <tool_request>{"tool_name": "run_code", "code": "print('git add .')", "language": "python"}</tool_request> 
    
    ----
    
    TOOL INTERACTIONS & WORKFLOWS:
    - Test code snippets before using write_file to save them
    - Use multiple run_code calls for iterative development and debugging
    - Combine with search tool for research-driven coding
    - Run validation/test code after writing files to ensure correctness
    - Use for actual GitHub operations - create repos, clone, push, pull, commit
    
    IMPORTANT: Never claim you "ran code" or "executed git commands" unless you actually made a <tool_request> with the run_code tool.
    
    CODE GUIDELINES:
    - Keep code under 100 lines for performance
    - Use "python" for calculations, algorithms, data processing
    - Use "shell" for basic system commands within sandbox AND for Git operations
    - Don't expect third-party libraries or network access (except GitHub via git/curl)
    - JSON must be valid - use double quotes
    - Always wrap in <tool_request></tool_request> tags
    - Don't hesitate to debug iteratively - multiple tool calls are encouraged
    - For any files you read or write (including code files in coding projects), always use the ./agent_scratch/ directory.
    - When asked to work with GitHub: ACTUALLY RUN the git commands using run_code, don't just print or simulate them
    - Always use the provided PAT for GitHub authentication
    ----- 
""".format(
    GITHUB_USERNAME=GITHUB_USERNAME,
    GITHUB_PAT=GITHUB_PAT,
    GITHUB_EMAIL=GITHUB_EMAIL,
    GITHUB_NOREPLY_EMAIL=GITHUB_NOREPLY_EMAIL
)

WRITE_FILE_TOOL_PROMPT = """
    ----- 
    WRITE FILE TOOL - Create or overwrite files safely within the agent sandbox directory.
    
    FORMAT: 
    <tool_request>
    {"tool_name": "write_file", "path": "filename.ext", "content": "file_content_here"}
    </tool_request>
    
    PARAMETERS:
    - path (required): Relative file path within agent sandbox (no absolute paths or directory traversal)
    - content (required): File content as string (max 1MB)
    
    EXAMPLES:
    
    Basic File Creation:
    <tool_request>
    {"tool_name": "write_file", "path": "script.py", "content": "print('Hello World')"}
    </tool_request>
    <tool_request>
    {"tool_name": "write_file", "path": "data.json", "content": "{\"name\": \"value\", \"number\": 42}"}
    </tool_request>
    <tool_request>
    {"tool_name": "write_file", "path": "README.md", "content": "# My Project\n\nThis is a sample project."}
    </tool_request>
    
    Nested Directory Creation:
    <tool_request>
    {"tool_name": "write_file", "path": "src/neural_network.py", "content": "class NeuralNet:\n    def __init__(self, layers):\n        self.layers = layers\n        self.weights = []\n    \n    def forward(self, x):\n        # Simple feedforward implementation\n        return x"}
    </tool_request>
    
    COMMON MISTAKES TO AVOID (NON-EXAMPLES):
    
    ❌ WRONG - Invalid tool_name name:
    <tool_request>{"tool_name": "create_file", "path": "file.py", "content": "code"}</tool_request>
    
    ❌ WRONG - Wrong field names:
    <tool_request>{"tool_name": "write_file", "filename": "file.py", "text": "code"}</tool_request>
    
    ❌ WRONG - Absolute paths:
    <tool_request>{"tool_name": "write_file", "path": "/home/user/file.py", "content": "code"}</tool_request>
    
    ❌ WRONG - Invalid JSON syntax (single quotes):
    <tool_request>{'tool_name': 'write_file', 'path': 'file.py', 'content': 'code'}</tool_request>
    
    ❌ WRONG - Missing content field:
    <tool_request>{"tool_name": "write_file", "path": "file.py"}</tool_request>
    
    ❌ WRONG - Unescaped double quotes in content:
    <tool_request>{"tool_name": "write_file", "path": "file.py", "content": "print("Hello")"}</tool_request>
    
    ❌ WRONG - Trailing comma in JSON:
    <tool_request>{"tool_name": "write_file", "path": "file.py", "content": "print('hi')",}</tool_request>
    
    ❌ WRONG - Newlines in content not properly escaped:
    <tool_request>{"tool_name": "write_file", "path": "file.py", "content": "line1
    line2"}</tool_request>
    
    ❌ WRONG - Backslash not properly escaped:
    <tool_request>{"tool_name": "write_file", "path": "file.py", "content": "C:\folder\file.txt"}</tool_request>
    
    SECURITY RESTRICTIONS:
    - Files created only within agent sandbox directory (./agent_scratch/)
    - Directory traversal attempts (../) are blocked
    - Maximum file size: 1MB
    - Automatically creates parent directories as needed
    - Completely overwrites existing files (no append mode)
    - Only allowed extensions: .txt, .md, .py, .sh, .json, .yaml, .yml, .html, .css, .js
    
    TOOL INTERACTIONS & WORKFLOWS:
    - Use read_file first to understand existing file structure
    - Combine with run_code to test code before writing files
    - Test written code with run_code after file creation
    
    IMPORTANT: Never claim you "wrote a file" unless you actually made a <tool_request> with the write_file tool.
    
    GUIDELINES:
    - Use relative paths only - no absolute paths or directory traversal
    - Ensure content is properly escaped for JSON
    - Parent directories are created automatically
    - Files are completely overwritten - use read_file first if preserving content
    - JSON must be valid - use double quotes
    - Always wrap in <tool_request></tool_request> tags
    - Tool name MUST be "write_file" with exact field names "path" and "content"
    - For any files you read or write (including code files in coding projects), always use the ./agent_scratch/ directory.
    ----- 
"""

READ_FILE_TOOL_PROMPT = """
    ----- 
    READ FILE TOOL - Read contents of existing files safely within the agent sandbox directory.
    
    FORMAT: 
    <tool_request>
    {"tool_name": "read_file", "path": "filename.ext"}
    </tool_request>
    
    PARAMETERS:
    - path (required): Relative file path within agent sandbox (no absolute paths or directory traversal)
    
    EXAMPLES:
    
    Basic File Reading:
    <tool_request>
    {"tool_name": "read_file", "path": "script.py"}
    </tool_request>
    <tool_request>
    {"tool_name": "read_file", "path": "data.json"}
    </tool_request>
    <tool_request>
    {"tool_name": "read_file", "path": "README.md"}
    </tool_request>
    
    Advanced Use Case:
    <tool_request>
    {"tool_name": "read_file", "path": "models/transformer_config.yaml"}
    </tool_request>
    
    COMMON MISTAKES TO AVOID (NON-EXAMPLES):
    
    ❌ WRONG - Invalid tool_name name:
    <tool_request>{"tool_name": "read_from_file", "path": "file.py"}</tool_request>
    
    ❌ WRONG - Wrong field names:
    <tool_request>{"tool_name": "read_file", "filename": "file.py"}</tool_request>
    
    ❌ WRONG - Absolute paths:
    <tool_request>{"tool_name": "read_file", "path": "/etc/passwd"}</tool_request>
    
    ❌ WRONG - Invalid JSON syntax (single quotes):
    <tool_request>{'tool_name': 'read_file', 'path': 'file.py'}</tool_request>
    
    ❌ WRONG - Missing path field:
    <tool_request>{"tool_name": "read_file"}</tool_request>
    
    ❌ WRONG - Unescaped double quotes in path:
    <tool_request>{"tool_name": "read_file", "path": "my"file.py"}</tool_request>
    
    ❌ WRONG - Trailing comma in JSON:
    <tool_request>{"tool_name": "read_file", "path": "file.py",}</tool_request>
    
    ❌ WRONG - Backslash not properly escaped:
    <tool_request>{"tool_name": "read_file", "path": "C:\folder\file.txt"}</tool_request>
    
    SECURITY RESTRICTIONS:
    - Files read only from agent sandbox directory (./agent_scratch/)
    - Directory traversal attempts (../) are blocked
    - Returns empty content if file doesn't exist or can't be read
    - UTF-8 encoding assumed for all files
    
    FILE EXISTENCE HANDLING:
    - If file exists: Returns full file content
    - If file doesn't exist: Returns empty string (no error)
    - Always check if returned content is empty to handle missing files
    
    TOOL INTERACTIONS & WORKFLOWS:
    - Use before write_file to understand existing code structure
    - Read multiple related files to understand project structure
    - Combine with search tool for understanding existing codebases
    - Use with run_code to read, test, and modify existing code iteratively
    
    IMPORTANT: Never claim you "read a file" unless you actually made a <tool_request> with the read_file tool.
    
    GUIDELINES:
    - Use relative paths only - no absolute paths or directory traversal
    - Check if returned content is empty (might indicate missing file)
    - JSON must be valid - use double quotes
    - Always wrap in <tool_request></tool_request> tags
    - Tool name MUST be "read_file" with exact field name "path"
    - Multiple read operations are fast and encouraged
    - For any files you read or write (including code files in coding projects), always use the ./agent_scratch/ directory.
    ----- 
"""